"""Add token_key to refresh_tokens for O(1) lookup.

Revision ID: 002
Revises: 001
Create Date: 2024-02-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    # Add token_key column for O(1) refresh token lookup
    op.add_column(
        'refresh_tokens',
        sa.Column('token_key', sa.String(length=64), nullable=True),
    )

    # Populate token_key for existing rows (sha256 of a placeholder — existing
    # tokens will need to be re-issued on next login, but the column is non-null-safe)
    op.execute("""
        UPDATE refresh_tokens
        SET token_key = md5(random()::text)
        WHERE token_key IS NULL
    """)

    # Make non-nullable after backfill
    op.alter_column('refresh_tokens', 'token_key', nullable=False)

    # Add unique index on token_key for O(1) lookup
    op.create_index('ix_refresh_tokens_token_key', 'refresh_tokens', ['token_key'], unique=True)

    # Drop the unique index on token_hash (lookup moved to token_key)
    op.drop_index('ix_refresh_tokens_token_hash', table_name='refresh_tokens')
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=False)

    # Add role column to users if missing
    op.add_column(
        'users',
        sa.Column('role', sa.String(length=20), nullable=True, server_default='user'),
    )

    # Backfill existing users
    op.execute("UPDATE users SET role = 'user' WHERE role IS NULL")
    op.alter_column('users', 'role', nullable=False, server_default='user')


def downgrade() -> None:
    # Remove role from users
    op.drop_column('users', 'role')

    # Restore unique index on token_hash
    op.drop_index('ix_refresh_tokens_token_hash', table_name='refresh_tokens')
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)

    # Drop token_key
    op.drop_index('ix_refresh_tokens_token_key', table_name='refresh_tokens')
    op.drop_column('refresh_tokens', 'token_key')
