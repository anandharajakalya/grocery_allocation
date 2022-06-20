"""
Utility for date and time
"""

import datetime
from datetime import datetime, timedelta


def get_today_date(format_required='%Y%m%d'):
    """
    Returns todays date
    :return: (str) todays date in specified format
    """
    today_date = datetime.today().strftime(format_required)
    return today_date


def add_days_to_date(days, date_string, format_required='%Y%m%d'):
    """

    :return:
    """
    date_after_add_days = (datetime.strptime(date_string, format_required) +
                           timedelta(days=days)).strftime(format_required)
    return date_after_add_days



