
import pytest
from datetime import datetime
from multi_agent_dev_team import AuthManager, Role

@pytest.fixture
def auth_config():
    return {
        "roles": {
            "admin": ["manage_users"],
            "developer": []
        },
        "agent_permissions": {}
    }

@pytest.fixture
def auth(auth_config):
    return AuthManager(auth_config["roles"], auth_config["agent_permissions"])

def test_create_user_by_admin(auth):
    admin_user_id = "admin_1"
    new_user = auth.create_user("test_user", Role.DEVELOPER, admin_user_id)
    assert new_user is not None
    assert new_user.username == "test_user"
    assert new_user.role == Role.DEVELOPER

def test_create_user_by_developer_fails(auth):
    admin_user_id = "admin_1"
    dev_user = auth.create_user("dev_user", Role.DEVELOPER, admin_user_id)
    with pytest.raises(PermissionError):
        auth.create_user("another_user", Role.DEVELOPER, dev_user.user_id)

def test_authenticate_valid_key(auth):
    admin_user_id = "admin_1"
    user = auth.authenticate(auth.users[admin_user_id].api_key)
    assert user is not None
    assert user.user_id == admin_user_id

def test_authenticate_invalid_key(auth):
    user = auth.authenticate("invalid_key")
    assert user is None
