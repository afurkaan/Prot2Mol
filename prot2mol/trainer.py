from transformers import Trainer, GenerationConfig
import torch

class GPT2_w_crs_attn_Trainer(Trainer):
    """
    Custom trainer for the Prot2Mol model.
    
    This trainer works with the unified Prot2MolModel that combines
    protein encoder and molecule decoder.
    """
    
    def __init__(self, *args, pchembl_only_mode=False, ignore_mismatched_optimizer=False, **kwargs):
        """
        Initialize the trainer with support for two-stage training.
        
        Args:
            pchembl_only_mode: If True, only train pChEMBL head (Stage 1)
            ignore_mismatched_optimizer: If True, skip loading optimizer state when it doesn't match
        """
        super().__init__(*args, **kwargs)
        self.pchembl_only_mode = pchembl_only_mode
        self.ignore_mismatched_optimizer = ignore_mismatched_optimizer
        # Storage for pChEMBL predictions during evaluation
        self.pchembl_predictions_list = []
        self.pchembl_targets_list = []
        
    def clear_pchembl_predictions(self):
        """Clear accumulated pChEMBL predictions. Called at the start of each evaluation."""
        self.pchembl_predictions_list = []
        self.pchembl_targets_list = []
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to clear pChEMBL predictions before evaluation starts.
        This prevents accumulation of predictions from previous evaluations.
        """
        # Clear any accumulated predictions from previous evaluations
        self.clear_pchembl_predictions()
        
        # Call parent evaluate
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss using the unified model with auxiliary pChEMBL prediction.
        
        Important:
        - LM loss is computed only for positive samples (train_lm=True)
        - pChEMBL loss is computed for ALL samples (both positive and negative)
        
        Args:
            model: The Prot2MolModel instance
            inputs: Dictionary containing input data
            return_outputs: Whether to return outputs along with loss
            
        Returns:
            Loss tensor, optionally with model outputs
        """
        # Extract inputs
        mol_input_ids = inputs["mol_input_ids"]
        prot_input_ids = inputs["prot_input_ids"]
        prot_attention_mask = inputs["prot_attention_mask"]
        
        # Use labels if provided, otherwise use mol_input_ids
        labels = inputs.get("labels", mol_input_ids)
        
        # Extract pChEMBL values and training flags
        pchembl_values = inputs.get("pchembl_values", None)
        train_lm = inputs.get("train_lm", True)  # Per-sample flags: True for positive, False for negative samples
        
        # Forward pass through the unified model
        outputs = model(
            mol_input_ids=mol_input_ids,
            prot_input_ids=prot_input_ids,
            prot_attention_mask=prot_attention_mask,
            labels=labels,
            pchembl_values=pchembl_values,
            train_lm=train_lm,
            pchembl_only_mode=self.pchembl_only_mode
        )
        
        loss = outputs["loss"]
        outputs["inputs"] = inputs
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step with auxiliary pChEMBL prediction.
        
        Override to handle model outputs that contain non-tensor values (dicts, booleans, etc.)
        which cannot be padded across processes in DDP.
        
        Important: pChEMBL predictions are stored for ALL samples, regardless of whether
        they are positive (train_lm=True) or negative (train_lm=False).
        
        Args:
            model: The Prot2MolModel instance
            inputs: Dictionary containing input data
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in the inputs
            
        Returns:
            Tuple of (loss, logits, labels) - all tensors or None
        """
        # Configure model for prediction
        model_config = model.config
        original_bos_token_id = model_config.bos_token_id
        model.config.bos_token_id = 1
        model.config.pad_token_id = 1
        model.config.forced_bos_token_id = 1
        model.config.forced_eos_token_id = 2
        
        try:
            has_labels = "labels" in inputs
            
            # Prepare model inputs
            # NOTE: train_lm should be a tensor from tokenization, not a Python boolean
            train_lm = inputs.get("train_lm", None)
            if train_lm is None:
                # Create a tensor of all True values if not provided
                batch_size = inputs["mol_input_ids"].shape[0]
                train_lm = torch.ones(batch_size, dtype=torch.bool, device=inputs["mol_input_ids"].device)
            
            model_inputs = {
                "mol_input_ids": inputs["mol_input_ids"],
                "prot_input_ids": inputs["prot_input_ids"],
                "prot_attention_mask": inputs["prot_attention_mask"],
                "labels": inputs.get("labels", inputs["mol_input_ids"]),
                "pchembl_values": inputs.get("pchembl_values", None),
                "train_lm": train_lm,
                "pchembl_only_mode": self.pchembl_only_mode
            }
            
            # Forward pass through model
            with torch.no_grad():
                outputs = model(**model_inputs)
            
            # Extract only tensor outputs to avoid padding issues
            # The model returns a dict, but we need loss, logits, and pchembl predictions
            if isinstance(outputs, dict):
                loss = outputs.get("loss", None)
                logits = outputs.get("logits", None)
                pchembl_preds = outputs.get("pchembl_predictions", None)
                
                # Store pChEMBL predictions for ALL samples (positive and negative)
                # We do this here because we can't return them from prediction_step (DDP padding issues)
                # Note: We store predictions regardless of train_lm flag - we want metrics for all samples
                if pchembl_preds is not None and model_inputs.get("pchembl_values") is not None:
                    self.pchembl_predictions_list.append(pchembl_preds.detach().cpu())
                    self.pchembl_targets_list.append(model_inputs["pchembl_values"].detach().cpu())
                
                # Debug: check loss value
                if loss is not None:
                    if torch.isnan(loss).any():
                        print(f"WARNING: NaN loss detected in prediction_step!")
                    if torch.isinf(loss).any():
                        print(f"WARNING: Inf loss detected in prediction_step!")
            else:
                # Handle tuple outputs
                loss = outputs[0] if len(outputs) > 0 else None
                logits = outputs[1] if len(outputs) > 1 else None
                pchembl_preds = None
            
            # Prepare labels for return
            if has_labels:
                labels = inputs["labels"]
            else:
                labels = None
            
            # If only loss is needed, return just that
            if prediction_loss_only:
                return (loss, None, None)
            
            # Return tensors only (no dicts, booleans, or other non-tensor types)
            # Note: We can't return pchembl_preds here because it would be padded across processes
            # causing the same error. We'll need to handle pChEMBL metrics differently.
            return (loss, logits, labels)
            
        finally:
            # Restore original bos_token_id
            model.config.bos_token_id = original_bos_token_id
    
    def get_pchembl_predictions(self, gather_across_ranks=True):
        """
        Retrieve accumulated pChEMBL predictions and targets.
        
        Args:
            gather_across_ranks: If True and in DDP mode, gather predictions from all ranks
                               to match what Transformers does with LM predictions
        
        Returns:
            Tuple of (predictions, targets) as numpy arrays, or (None, None) if empty
        """
        if len(self.pchembl_predictions_list) == 0:
            return None, None
        
        # Concatenate all batches on this rank
        predictions = torch.cat(self.pchembl_predictions_list, dim=0)
        targets = torch.cat(self.pchembl_targets_list, dim=0)
        
        local_samples = predictions.shape[0]
        
        # Gather across ranks if requested and in DDP mode
        # NOTE: Transformers gathers LM predictions before calling compute_metrics,
        # so we need to do the same for pChEMBL predictions to ensure counts match
        if gather_across_ranks and self.args.local_rank != -1:
            import torch.distributed as dist
            import logging
            logger = logging.getLogger(__name__)
            
            # Move to CUDA for gathering
            device = torch.device(f"cuda:{self.args.local_rank}")
            predictions = predictions.to(device)
            targets = targets.to(device)
            
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            # Gather sizes first
            local_size = torch.tensor([predictions.shape[0]], device=device)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            
            max_size = max([s.item() for s in size_list])
            
            # Pad if needed
            if predictions.shape[0] < max_size:
                padding = torch.zeros(max_size - predictions.shape[0], device=predictions.device)
                predictions = torch.cat([predictions, padding], dim=0)
                targets = torch.cat([targets, padding], dim=0)
            
            # Gather from all ranks
            gathered_preds = [torch.zeros_like(predictions) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(targets) for _ in range(world_size)]
            
            dist.all_gather(gathered_preds, predictions)
            dist.all_gather(gathered_targets, targets)
            
            # Concatenate and trim
            predictions_list = []
            targets_list = []
            for i, size in enumerate(size_list):
                predictions_list.append(gathered_preds[i][:size.item()])
                targets_list.append(gathered_targets[i][:size.item()])
            
            predictions = torch.cat(predictions_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            
            total_samples = predictions.shape[0]
           
        else:
            if self.args.local_rank != -1:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Rank {self.args.local_rank}: Returning {local_samples} LOCAL pChEMBL predictions (no gathering)")
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        return predictions_np, targets_np
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """
        Override to handle mismatched optimizer states gracefully.
        
        When model architecture changes (e.g., different number of layers),
        the optimizer state won't match. This method catches that error and
        starts with a fresh optimizer instead.
        """
        if not self.ignore_mismatched_optimizer:
            # Use default behavior
            return super()._load_optimizer_and_scheduler(checkpoint)
        
        try:
            # Try to load optimizer state
            super()._load_optimizer_and_scheduler(checkpoint)
        except (ValueError, RuntimeError) as e:
            if "doesn't match the size" in str(e) or "size mismatch" in str(e):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Optimizer state doesn't match current model architecture. "
                    f"Starting with fresh optimizer. Error: {e}"
                )
                # Don't load optimizer/scheduler state - will use freshly initialized ones
                # The parent class already initialized them in __init__
            else:
                # Re-raise if it's a different error
                raise
