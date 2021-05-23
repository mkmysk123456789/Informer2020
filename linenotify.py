import requests


def send_line_notify():
    """
    LINEに通知する
    """
    line_notify_token = 'AqDWTPO25wqUrP8qSosG9JBpuwKhTKvmf8AUyg99Bh4'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: finished'}
    requests.post(line_notify_api, headers=headers, data=data)
