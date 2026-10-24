from django.conf import settings


DEMO_ACCOUNT_GROUP = "BabyLog demo accounts"
DEMO_ACCOUNT_USERNAMES = {
    "short": "babylog-demo-short",
    "long": "babylog-demo-long",
}
DEMO_SEED_PREFIX = "demo-seed:v1:"


def is_demo_account(user):
    return (
        user.is_authenticated
        and user.username in DEMO_ACCOUNT_USERNAMES.values()
        and user.groups.filter(name=DEMO_ACCOUNT_GROUP).exists()
    )


def demo_context(request):
    return {
        "is_demo_account": is_demo_account(request.user),
        "demo_accounts_enabled": settings.DEMO_ACCOUNTS_ENABLED,
    }
