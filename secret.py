import secrets

# Generate a random secret key
secret_key = secrets.token_hex(16)
print("Generated secret key:", secret_key)
