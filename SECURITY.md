# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by:

1. **Do not** open a public GitHub issue
2. Email security concerns to the maintainers
3. Include details about the vulnerability
4. Allow time for a fix before public disclosure

## Security Considerations

This repository contains educational examples for fine-tuning ML models. When using these examples:

- **API Keys**: Never commit API keys or tokens to version control
- **Model Downloads**: Only download models from trusted sources (official Hugging Face Hub)
- **Data Privacy**: Be cautious with sensitive data in training datasets
- **Compute Resources**: Monitor GPU/CPU usage during training

## Best Practices

1. Use environment variables for sensitive configuration
2. Review model licenses before commercial use
3. Validate input data before training
4. Keep dependencies updated for security patches
