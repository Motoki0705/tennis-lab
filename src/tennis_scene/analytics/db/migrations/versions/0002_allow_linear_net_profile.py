"""Allow the standard linear hard-court net profile."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    name = op.f("ck_court_net_profiles_net_profile_kind")
    op.drop_constraint(name, "court_net_profiles", type_="check")
    op.create_check_constraint(
        name,
        "court_net_profiles",
        "net_profile_kind IN ('flat', 'itf_linear', 'itf_parabolic', 'measured_curve')",
    )


def downgrade() -> None:
    name = op.f("ck_court_net_profiles_net_profile_kind")
    op.drop_constraint(name, "court_net_profiles", type_="check")
    op.create_check_constraint(
        name,
        "court_net_profiles",
        "net_profile_kind IN ('flat', 'itf_parabolic', 'measured_curve')",
    )
