  # Dataset Format
    
    example_dataset/
    ├── data_0/
    │   ├── input_images/
    │   │   ├── 0.jpg (wrist view at t-1)
    │   │   ├── 1.jpg (wrist view at t)
    │   │   ├── 2.jpg (agent view at t-1)
    │   │   └── 3.jpg (agent view at t)
    │   ├── text.txt (task description)
    │   └── target.jpg (agentview at t+8)
    
# Output Collated Batch

      - pixel_values: dict
          - dino: torch.Tensor (bsz, history_img_seq_len, 3, 224, 224)
          - siglip: torch.Tensor (bsz, history_img_seq_len, 3, 224, 224)
      - input_ids: torch.Tensor (bsz, seq_len)
      - [dummy] attention_mask: torch.Tensor (bsz, seq_len)
      - [dummy] labels: torch.Tensor (bsz, seq_len)
      - goal_image: torch.Tensor (bsz, 3, 224, 224)
      - [dummy] actions: torch.Tensor (bsz, action_horizon, action_dim)
      - [dummy] proprios: torch.Tensor (bsz, history_horizon, 7) # if use wrist, history_img_seq_len = 2 * history_horizon
      - [dummy] future_wrist_view: torch.Tensor (bsz, 3, 224, 224)
      - [dummy] language_planning_input_ids: not involved in the example dataset
      - [dummy] language_planning_attention_mask: not involved in the example dataset
      - [dummy] language_planning_labels: not involved in the example dataset
