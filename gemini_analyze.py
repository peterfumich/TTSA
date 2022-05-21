import time
import requests

def Price_Data(time_step,symbol):
    base_url = "https://api.gemini.com/v2"
    fail = True
    while fail == True:
        try:
            response = requests.get(base_url + "/candles/" + symbol + "/" + time_step)
            fail = False
        except:
            print("Failed to requests.get candlestick data. Trying again in one minute. ")
            time.sleep(60)
            fail = True
    try:
        candle_data = response.json()
        #print(candle_data)
        prices = [x[4] for x in candle_data]  # 1 for open data 4 for closing data
        candle_volume = [x[-1] for x in candle_data]
        return(prices,candle_volume)#ADDED THIS FOR THE SANDBOX-JUST WANTED THE PRICES AND VOLUMES, NOT THE AVERAGING ARRAYS
    except:
        return([0,0],0)
