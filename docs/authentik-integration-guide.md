# Authentik Integration Guide for Open WebUI

This guide provides step-by-step instructions for integrating Authentik as the identity provider for Open WebUI using OpenID Connect (OIDC).

## Overview

Open WebUI has built-in OAuth2/OIDC support that works seamlessly with Authentik. This integration allows you to:

- Use Authentik as the authoritative identity provider
- Enable single sign-on (SSO) for users
- Maintain multi-tenant isolation (users only see their own data)
- Support role and group-based access control

## Prerequisites

- Running Authentik instance (Docker or other deployment)
- Open WebUI deployment
- Admin access to both systems

## Step 1: Configure Authentik

### 1.1 Create OAuth2/OpenID Connect Provider

1. Log in to your Authentik admin interface
2. Navigate to **Applications → Providers**
3. Click **Create** and select **OAuth2/OIDC Provider**
4. Configure the provider:

   **Basic Settings:**
   - **Name**: `open-webui` (or your preferred name)
   - **Authentication flow**: `Authorization Code`
   - **Client ID**: `open-webui-client` (or generate a unique ID)
   - **Client Secret**: Generate and save this securely
   - **Redirect URIs/Origins**: `https://your-openwebui-host/oauth/oidc/callback`

   **Advanced Settings:**
   - **Signing Key**: Select your signing key
   - **Scopes**: Enable `openid`, `email`, `profile`
   - **PKCE**: Enable (recommended for security)

   **Protocol Settings:**
   - **Subject mode**: `users` (recommended) or `based on user's ID`
   - **User info method**: `scope claims`

### 1.2 Create Application

1. Navigate to **Applications → Applications**
2. Click **Create**
3. **Name**: `Open WebUI`
4. **Slug**: `open-webui`
5. **Provider**: Select the provider you just created
6. **Launch URL**: `https://your-openwebui-host`
7. **Redirect URI**: Should be auto-populated from provider settings
8. **Access Policy**: Set appropriate access (e.g., Default allow or group-based)

### 1.3 Note Important URLs

After creating the provider and application, note these URLs:
- **Client ID**: From the provider configuration
- **Client Secret**: From the provider configuration
- **Discovery URL**: `https://authentik.company/application/o/open-webui/.well-known/openid-configuration`

## Step 2: Configure Open WebUI

### 2.1 Environment Variables

Add the following configuration to your `.env` file:

```bash
# Enable OAuth/OIDC authentication
ENABLE_OAUTH_SIGNUP=true

# Authentik OAuth2/OIDC Configuration
OAUTH_CLIENT_ID=your_authentik_client_id_here
OAUTH_CLIENT_SECRET=your_authentik_client_secret_here
OAUTH_PROVIDER_NAME=authentik

# OpenID Connect Configuration
OPENID_PROVIDER_URL=https://authentik.company/application/o/open-webui/.well-known/openid-configuration
OPENID_REDIRECT_URI=https://your-openwebui-host/oauth/oidc/callback

# Optional: Merge existing accounts by email (recommended for migration)
OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true

# OAuth Scopes
OAUTH_SCOPES="openid email profile"

# PKCE Configuration (recommended for security)
OAUTH_CODE_CHALLENGE_METHOD=S256
```

### 2.2 Optional Configuration

For role and group management:

```bash
# Enable role management from Authentik
ENABLE_OAUTH_ROLE_MANAGEMENT=false

# Enable group management from Authentik
ENABLE_OAUTH_GROUP_MANAGEMENT=false

# Role mappings (if using role management)
OAUTH_ROLES_CLAIM=roles
OAUTH_ALLOWED_ROLES=user,admin
OAUTH_ADMIN_ROLES=admin

# Group mappings (if using group management)
OAUTH_GROUPS_CLAIM=groups
OAUTH_BLOCKED_GROUPS=blocked-group
```

### 2.3 Claim Mappings

Open WebUI already has sensible defaults for Authentik claims:

- **Subject Claim**: `sub` (unique identifier)
- **Username Claim**: `preferred_username`
- **Email Claim**: `email`
- **Picture Claim**: `picture`

These can be overridden if your Authentik setup uses different claim names.

## Step 3: Verify Configuration

### 3.1 Check OpenID Configuration

Test that your Authentik discovery endpoint is accessible:

```bash
curl https://authentik.company/application/o/open-webui/.well-known/openid-configuration
```

### 3.2 Check Open WebUI OAuth Providers

Start Open WebUI and check the `/api/config` endpoint to verify the OIDC provider is loaded:

```bash
curl http://localhost:8000/api/config | jq '.oauth.providers'
```

### 3.3 Test Authentication Flow

1. Navigate to your Open WebUI instance
2. You should see a "Continue with SSO" button (customized to your provider name)
3. Click the button to initiate OAuth flow
4. You should be redirected to Authentik for authentication
5. After successful authentication, you'll be redirected back to Open WebUI

## Step 4: Advanced Configuration

### 4.1 Role and Group Management

To enable role-based access control from Authentik:

1. **In Authentik**:
   - Configure group mappings or property mappings for roles
   - Add roles/groups to user tokens

2. **In Open WebUI**:
   ```bash
   ENABLE_OAUTH_ROLE_MANAGEMENT=true
   ENABLE_OAUTH_GROUP_MANAGEMENT=true
   ```

### 4.2 Custom Claim Mappings

If Authentik uses non-standard claim names, configure them in Open WebUI:

```bash
OAUTH_SUB_CLAIM=sub
OAUTH_USERNAME_CLAIM=preferred_username
OAUTH_EMAIL_CLAIM=email
OAUTH_PICTURE_CLAIM=picture
OAUTH_ROLES_CLAIM=roles
OAUTH_GROUPS_CLAIM=groups
```

### 4.3 Multi-tenant Isolation

Open WebUI automatically isolates user data by:
- Using the OAuth `sub` claim as the unique user identifier
- Associating all chats, files, and settings with the user ID
- Enforcing access controls at the API level

## Step 5: Security Considerations

### 5.1 Recommended Settings

- **PKCE**: Always enable PKCE (`OAUTH_CODE_CHALLENGE_METHOD=S256`)
- **HTTPS**: Use HTTPS in production
- **Token Validation**: Ensure proper token validation is enabled
- **Session Management**: Configure appropriate session timeouts

### 5.2 Network Security

- Firewall rules to restrict access to Authentik
- Use reverse proxy with proper SSL termination
- Enable CORS restrictions appropriately

## Step 6: Troubleshooting

### 6.1 Common Issues

**Authentication not working**:
- Verify `OPENID_PROVIDER_URL` is correct and accessible
- Check Client ID and Secret match Authentik configuration
- Ensure redirect URI matches exactly (including protocol)

**User not created after login**:
- Check `ENABLE_OAUTH_SIGNUP=true`
- Verify `OAUTH_MERGE_ACCOUNTS_BY_EMAIL` setting
- Check server logs for error messages

**Logout not working**:
- Ensure `OPENID_PROVIDER_URL` is set for end-session discovery
- Check if Authentik end-session endpoint is accessible

### 6.2 Debugging

Enable debug logging to troubleshoot issues:

```bash
# Add to environment variables
SRC_LOG_LEVELS="OAUTH=DEBUG"
```

Check Open WebUI logs for OAuth flow details.

## Step 7: Migration Strategy

### 7.1 Existing Users

To migrate existing local users to Authentik:

1. Set `OAUTH_MERGE_ACCOUNTS_BY_EMAIL=true`
2. Users should use the same email in Authentik as their local account
3. First login via Authentik will link to existing account
4. Local login can be disabled after migration

### 7.2 Rollback Plan

To revert to local authentication:

1. Remove or comment out OAuth environment variables
2. Set `WEBUI_AUTH=true` for local authentication
3. Users can continue with existing local accounts

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_OAUTH_SIGNUP` | `false` | Enable user creation via OAuth |
| `OAUTH_CLIENT_ID` | - | OAuth client ID from Authentik |
| `OAUTH_CLIENT_SECRET` | - | OAuth client secret from Authentik |
| `OAUTH_PROVIDER_NAME` | `SSO` | Display name for OAuth provider |
| `OPENID_PROVIDER_URL` | - | Authentik OpenID discovery URL |
| `OPENID_REDIRECT_URI` | - | OAuth callback URL |
| `OAUTH_MERGE_ACCOUNTS_BY_EMAIL` | `false` | Merge accounts by email |
| `OAUTH_SCOPES` | - | OAuth scopes (default: openid email profile) |
| `OAUTH_CODE_CHALLENGE_METHOD` | - | PKCE method (recommended: S256) |

### Authentik URLs

- **Discovery URL**: `https://authentik.company/application/o/{slug}/.well-known/openid-configuration`
- **Authorization**: `https://authentik.company/application/o/authorize/`
- **Token**: `https://authentik.company/application/o/token/`
- **User Info**: `https://authentik.company/application/o/userinfo/`
- **End Session**: `https://authentik.company/application/o/{slug}/end-session/`

## Support

For issues or questions:
1. Check Open WebUI server logs
2. Verify Authentik application logs
3. Ensure network connectivity between services
4. Review environment variable configuration

This integration provides a secure, standards-based authentication solution using Authentik as the identity provider for Open WebUI.