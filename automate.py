import time
from datetime import datetime
import os
import cryptpandas
import numpy as np
import matplotlib.pyplot as plt

# import scipy as sp
import pandas as pd
import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from strategy import wrapper


def task():
    """Task to be executed every 19 minutes."""

    plt.rcParams["figure.figsize"] = (16, 8)
    plt.rcParams["figure.dpi"] = 300

    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["legend.title_fontsize"] = 20
    plt.rcParams["figure.titlesize"] = 20

    plt.rcParams["font.family"] = "Times New Roman"

    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "data")

    # all_existing_files = os.listdir(data_dir)

    releases = [
        "release_3547",
        "release_3611",
        "release_3675",
        "release_3739",
        "release_3803",
        "release_3867",
    ]
    passwords = [
        "oUFtGMsMEEyPCCP6",
        "GMJVDf4WWzsV1hfL",
        "PSI9bPh4aM3iQMuE",
        "1vA9LaAZDTEKPePs",
        "0n74wuaJ2wm8A4qC",
        "mXTi0PZ5oL731Zqx",
    ]

    # Replace with your OAuth token
    slack_token = "xoxb-8020284472341-8025452276167-g5PLFEJ9GgRLxpEosg010G9B"

    # Initialize Slack client
    client = WebClient(token=slack_token)

    # Replace with the channel ID you want to read messages from
    channel_id = "C080P6M4DKL"

    def fetch_messages(channel_id, limit=10000):
        try:
            # Fetch messages from the channel
            response = client.conversations_history(channel=channel_id, limit=limit)
            # for message in response.get("messages", []):
            #     print(f"User: {message.get('user', 'N/A')} | Text: {message.get('text')}")

            return response.get("messages", [])

        except SlackApiError as e:
            print(f"Error fetching messages: {e.response['error']}")

    def fetch_passcodes(messages):
        messages = messages[::-1]
        for msg in messages:

            if msg["user"] == "U080GCRATP1" and msg["type"] == "message":
                text = msg.get("text")
                str_to_find = "Data has just been released"
                if str_to_find in text:
                    # print("Found")
                    # print(text)
                    words = text.split(" ")
                    for word in words:
                        if word.endswith(".crypt'") and word.startswith("'"):
                            rel_num = word[1 : word.index(".crypt")]
                            if rel_num not in releases:
                                releases.append(rel_num)
                            else:
                                pass
                            # print(rel_num)
                            # break
                        if word == "passcode":
                            passcode = words[words.index(word) + 2][1:-2]
                            if passcode not in passwords:
                                passwords.append(passcode)
                            else:
                                pass
            else:
                pass
        return releases, passwords

    def time_to_mins(time):
        datetime_obj = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M")
        seconds_since_epoch = int(datetime_obj.timestamp())
        return seconds_since_epoch

    # Fetch the latest messages
    messages = fetch_messages(channel_id)

    times = []
    with open(os.path.join(os.getcwd(), "date_and_times.txt"), "r") as f:
        # f.write(str(datetime.datetime.now()))
        dates = f.readlines()
        for date in dates:
            time = date[3 : date.rfind(":")]
            times.append(time)

    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    for i, time in enumerate(times[:-1]):
        # print(time, type(time))
        int_time = time_to_mins(time)
        int_time_now = time_to_mins(time_now)
        next_time = time_to_mins(times[i + 1])
        int_next_time = time_to_mins(times[i + 1])

        if int_time <= int_time_now and int_time_now >= int_next_time:
            releases, passwords = fetch_passcodes(fetch_messages(channel_id))
            print("Found passcodes")
            print(len(releases), len(passwords))
            break

    final_rel = releases[-1]
    final_pass = passwords[-1]

    print("final_release and final_pass: ", final_rel, final_pass)

    df = cryptpandas.read_encrypted(
        path=os.path.join(data_dir, f"{final_rel}.crypt"),
        password=final_pass,
    )

    portfolio = wrapper(df)
    print(portfolio)
    # print(fetch_passcodes(fetch_messages(channel_id)))

    # release_times = np.loadtxt(os.path.join(os.getcwd(), "date_and_times.txt"), dtype=str)
    # print(release_times)
    print(f"Task executed at: {datetime.now()}")


def main():
    while True:
        task()  # Execute your task
        time.sleep(5)  # Wait for 19 minutes (19 minutes * 60 seconds)


if __name__ == "__main__":
    main()
