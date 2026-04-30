# TinyAD2 VisualAD TinyCLIP Support

## Backbone configuration
- Default backbone settings live in `configs/backbone_settings.yaml`.
- Switch to TinyCLIP by name and provide weights via URL or local path:
  - `--backbone TinyCLIP-ViT-L/14@336px --backbone_weights /path/to/tinyclip.pt`
- The config provides `embed_dim`, `transformer_layers`, `image_size`, and `layers` defaults.
- CLI overrides are available: `--embed_dim`, `--transformer_layers`, `--image_size`, `--features_list`.

## Fine-tuning
- Unfreeze the last N vision blocks with a small LR:
  - `--unfreeze_encoder_layers 2 --encoder_learning_rate 1e-5`

## Dry-run test
Run the minimal forward/loss check:
```
python -m unittest discover -s tests
```
