import requests


def send_line_notify(message='Just Finished Learning.'):
    """
    LINEに通知する
    """
    line_notify_token = 'yDrpFWiOnY1rgsNqBmAcTIKMPbUsMqicfVronvuGmtF'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': message}
    requests.post(line_notify_api, headers=headers, data=data)
