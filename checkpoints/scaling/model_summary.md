# Part 2: Model Architectures and Training Statistics

## Model Configurations

| Model | Params | d_model | Layers | Heads | d_ff |
|-------|--------|---------|--------|-------|------|
| tiny | 1,311,872 | 128 | 4 | 4 | 512 |
| small | 3,443,136 | 192 | 6 | 6 | 768 |
| medium | 8,654,208 | 384 | 4 | 6 | 1536 |
| large | 20,978,176 | 512 | 6 | 8 | 2048 |
| xl | 45,623,040 | 768 | 6 | 12 | 3072 |

## Training Results

| Model | Params | Val Loss | Train Loss | Time (s) | Tok/s | GPU MB |
|-------|--------|----------|------------|----------|-------|--------|
| tiny | 1,311,872 | 1.7868 | 1.7962 | 231 | 636,807 | 4,534 |
| small | 3,443,136 | 1.6252 | 1.6219 | 445 | 331,244 | 4,534 |
| medium | 8,654,208 | 1.4835 | 1.4885 | 549 | 268,258 | 4,534 |
| large | 20,978,176 | 1.3980 | 1.3932 | 1094 | 134,630 | 4,534 |
| xl | 45,623,040 | 1.3139 | 1.3304 | 1911 | 77,084 | 4,534 |

## Scaling Law Fit

**Power Law:** L = 19.7157 * N^(-0.2220) + 0.9235

- Scaling exponent alpha = 0.2220
- Irreducible loss c = 0.9235
- R-squared = 0.9989