# AWS Bedrock Implementation for HTTP Server

## Key Features Added

1. **Provider Selection**
   - Added support for both Anthropic API and AWS Bedrock
   - Configurable via environment variables or API requests
   - Default model names specific to each provider

2. **AWS Credential Validation**
   - Automatic validation of AWS credentials
   - Checks for boto3 installation, credentials availability, and AWS region setting
   - Graceful error handling for missing credentials

3. **Model Name Formatting**
   - Added format_model_name_for_provider function to ensure correct model format
   - Automatic mapping between Anthropic and Bedrock model naming conventions
   - Proper prefixing with "anthropic." for Bedrock models

4. **Dynamic Provider Switching**
   - API support for switching between providers at runtime
   - Intelligent model selection when changing providers
   - Preserves configuration across provider changes

5. **Configuration UI Updates**
   - Exposed provider selection in the configuration API
   - Available providers list in the configuration response
   - Model name validation specific to each provider

## How to Use

### With Environment Variables:
```
export API_PROVIDER=bedrock
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

### With API Configuration:
```
POST /api/config
{
  "api_provider": "bedrock",
  "model": "anthropic.claude-3-7-sonnet-20250219-v1:0"
}
```

### Docker Example:
```
docker run \
    -e API_PROVIDER=bedrock \
    -e AWS_REGION=us-west-2 \
    -e AWS_ACCESS_KEY_ID=your_key_id \
    -e AWS_SECRET_ACCESS_KEY=your_secret_key \
    -p 8083:8083 \
    -it your-image-name
```

## Implementation Details

1. The `ServerConfig` class has been updated to support both providers
2. A new `validate_aws_credentials()` function checks AWS credential validity
3. Model names are automatically formatted for the appropriate provider 
4. The API now accepts and returns the provider configuration
5. Error handling includes provider-specific validation

The implementation is complete and ready to use with AWS Bedrock.
