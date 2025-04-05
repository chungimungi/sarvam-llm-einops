# sarvam-llm-einops

## Overview
This library provides a flexible tensor rearrangement function inspired by einops, allowing you to manipulate tensor dimensions through a readable pattern-based syntax. It supports splitting axes, merging axes, and repeating axes in NumPy arrays.

## Core Functionality
The `rearrange` function transforms tensors according to pattern specifications using the syntax:
```python
rearrange(tensor, "src_pattern -> target_pattern", **axes_lengths)
```

## Features
- **Axis Splitting**: Convert a single dimension into multiple dimensions
- **Axis Merging**: Combine multiple dimensions into a single dimension
- **Axis Repetition**: Repeat specific axes by a given factor
- **Automatic Shape Inference**: Deduce missing dimension sizes when possible

## Design Decisions

### Pattern Syntax
- Uses an einops-inspired `src -> target` pattern for readability
- Supports grouped dimensions with parentheses: `(dim1 dim2)` for merging/splitting
- Explicit repetition syntax: `(repeat dim1 dim2)` with a required `repeat` factor

### Error Handling
- Comprehensive validation ensures tensor shapes match pattern specifications
- Detailed error messages identify specific issues (dimension mismatches, invalid syntax)
- Early validation to catch issues before tensor operations

### Dimension Inference
- Automatically infers single unknown dimensions in grouped axes
- Requires explicit dimensions for ambiguous cases

### Implementation Approach
1. Parse and validate the pattern syntax
2. Build a dimension mapping for all axis names
3. Split the tensor into individual axes
4. Apply any repeat operations
5. Permute the axes to match target order
6. Reshape to the final target configuration

## Usage Examples
```python
# Reshape a batch of images from (batch, height, width, channels) to (batch, channels, height, width)
rearrange(images, "b h w c -> b c h w")

# Split a dimension into parts
rearrange(tensor, "b (h h2) w -> b h h2 w", h2=2)

# Merge dimensions
rearrange(tensor, "b h w c -> b (h w) c")

# Repeat specific dimensions
rearrange(tensor, "b h w -> b (repeat h w)", repeat=2)
```

*NOTE: Readme generated with Claude3.7 & verified by me.*
