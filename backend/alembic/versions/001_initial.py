"""Create initial tables.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE exercisename AS ENUM ('squat', 'deadlift', 'bench_press', 'overhead_press', 'pull_up', 'push_up', 'lunge', 'plank', 'custom')")
    op.execute("CREATE TYPE analysisstatus AS ENUM ('pending', 'processing', 'completed', 'failed')")

    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Refresh tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('token_hash', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)

    # Analyses table
    op.create_table(
        'analyses',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('task_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('exercise_name', postgresql.ENUM('squat', 'deadlift', 'bench_press', 'overhead_press', 'pull_up', 'push_up', 'lunge', 'plank', 'custom', name='exercisename'), nullable=False),
        sa.Column('trainer_video_url', sa.String(length=500), nullable=True),
        sa.Column('user_video_url', sa.String(length=500), nullable=False),
        sa.Column('trainer_video_key', sa.String(length=255), nullable=True),
        sa.Column('user_video_key', sa.String(length=255), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'processing', 'completed', 'failed', name='analysisstatus'), nullable=False, default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, default=0),
        sa.Column('current_stage', sa.String(length=100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('reps', sa.Integer(), nullable=True),
        sa.Column('feedback_summary', sa.Text(), nullable=True),
        sa.Column('technical_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('frames_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('task_id')
    )
    op.create_index('ix_analyses_task_id', 'analyses', ['task_id'], unique=True)
    op.create_index('ix_analyses_user_id', 'analyses', ['user_id'])
    op.create_index('ix_analyses_status', 'analyses', ['status'])
    op.create_index('ix_analyses_created_at', 'analyses', ['created_at'])

    # Analysis frames table
    op.create_table(
        'analysis_frames',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('analysis_id', sa.UUID(), nullable=False),
        sa.Column('frame_id', sa.Integer(), nullable=False),
        sa.Column('error_score', sa.Float(), nullable=False),
        sa.Column('feedback', sa.Text(), nullable=True),
        sa.Column('technical_observation', sa.Text(), nullable=True),
        sa.Column('user_image_key', sa.String(length=255), nullable=True),
        sa.Column('trainer_image_key', sa.String(length=255), nullable=True),
        sa.Column('joint_angles', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('pose_landmarks', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_analysis_frames_analysis_id', 'analysis_frames', ['analysis_id'])
    op.create_index('ix_analysis_frames_frame_id', 'analysis_frames', ['analysis_id', 'frame_id'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('ix_analysis_frames_frame_id', table_name='analysis_frames')
    op.drop_index('ix_analysis_frames_analysis_id', table_name='analysis_frames')
    op.drop_table('analysis_frames')

    op.drop_index('ix_analyses_created_at', table_name='analyses')
    op.drop_index('ix_analyses_status', table_name='analyses')
    op.drop_index('ix_analyses_user_id', table_name='analyses')
    op.drop_index('ix_analyses_task_id', table_name='analyses')
    op.drop_table('analyses')

    op.drop_index('ix_refresh_tokens_token_hash', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_user_id', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')

    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS analysisstatus")
    op.execute("DROP TYPE IF EXISTS exercisename")