import apprise


def get_apobj():
    apobj = apprise.Apprise()
    with open("../../discord_credentials.txt") as fp:
        webhook_path = fp.readline()
    apobj.add(webhook_path)

    return apobj
