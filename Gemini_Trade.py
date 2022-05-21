import time
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests, json
import base64
import hmac
import hashlib
#import gdrive
import os
import tkinter as tk
#API KEY
import gemini_analyze
import websocket

root_directory = os.getcwd()
keys_dict = {'gemini_api_key':'','gemini_api_secret':''}
ID = {'id':''}
#user_id = "123456789"
def Set_Balances(*args):
    if str(key.get()) != '':
        base_url = "https://api.gemini.com"
        endpoint = "/v1/notionalbalances/usd"

        url = base_url + endpoint
        t = datetime.datetime.now()
        payload_nonce = str(int(time.mktime(t.timetuple()) * 1000)) + str(np.random.randint(1000, 10000))
        payload = {"nonce": int(payload_nonce), "request": "/v1/notionalbalances/usd"}
        encoded_payload = json.dumps(payload).encode()
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(keys_dict['gemini_api_secret'].encode(), b64, hashlib.sha384).hexdigest()
        request_headers = {'Content-Type': "text/plain", 'Content-Length': "0", 'X-GEMINI-APIKEY': keys_dict['gemini_api_key'],
                           'X-GEMINI-PAYLOAD': b64, 'X-GEMINI-SIGNATURE': signature, 'Cache-Control': "no-cache"}
        try:
            response = requests.post(url, data=None, headers=request_headers)
            balances = response.json()
        except:
            usdbalance.set(0)
        #print(balances)
        for account in balances:
            if account["currency"]=="USD":
                usdbalance.set(account["amount"])
            elif account["currency"]==str(symbol_entry.get()).upper():
                symbolbalance.set(account["amount"])
def update_tradetotal(*args):
    if str(price_entry.get())!='' and str(amount_entry.get())!='':
        tradetotal.set(str(float(price_entry.get()) * float(amount_entry.get())))
    else:
        tradetotal.set(0)
def Set_User():

    ID['id'] = str(user_id.get())
    key_path = os.path.join(root_directory, ID['id'] + "/keys.txt")
    with open(key_path, 'r') as file:
        keys = file.readlines()
        keys_dict['gemini_api_key'] = str(keys[0][:-1])
        keys_dict['gemini_api_secret'] = str(keys[1][:-1])#.encode()
    key.set(keys_dict['gemini_api_key'])
    label_symbolbalance.set(str(symbol_entry.get())+"Balance")

    Set_Balances()
value_storage = "usd"
#
def Gemini_Trade(symbol, amount, price, side,trade_type, gemini_api_key, gemini_api_secret):
    gemini_api_secret = gemini_api_secret.encode()
    base_url = "https://api.gemini.com"
    endpoint = "/v1/order/new"
    url = base_url + endpoint
    t = datetime.datetime.now()
    payload_nonce = str(int(time.mktime(t.timetuple()) * 1000)) + str(np.random.randint(1000, 10000))
    payload = {
        "request": "/v1/order/new",
        "nonce": payload_nonce,
        "symbol": symbol + value_storage,
        "amount": str(amount),
        "price": str(price),
        "side": side,
        "type": trade_type
        # ,
        # "options": ["maker-or-cancel"]

    }
    #REALLY NEED TO MAKE SURE WE ARE DOING THE RIGHT THING HERE HAVE A VERIFICATION STEP, LIKE ARE YOU REALLY FUCKING
    #SURE YOU WANT TO YOLO.

    encoded_payload = json.dumps(payload).encode()
    b64 = base64.b64encode(encoded_payload)
    signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

    request_headers = {'Content-Type': "text/plain",
                       'Content-Length': "0",
                       'X-GEMINI-APIKEY': gemini_api_key,
                       'X-GEMINI-PAYLOAD': b64,
                       'X-GEMINI-SIGNATURE': signature,
                       'Cache-Control': "no-cache"}

    response = requests.post(url,
                             data=None,
                             headers=request_headers)
    new_order = response.json()
    results.set(new_order)
    print(new_order)
    Set_Balances()
    return (new_order)

def display_data():
    Y = gemini_analyze.Price_Data(timestep.get(), symbol_entry.get() + value_storage)[0]
    Y = Y[0:int(timerange_entry.get())]
    Y = Y[::-1]
    plt.plot(Y)
    plt.savefig('price_data.png', )
    plt.clf()
    img = (Image.open("price_data.png"))
    image = ImageTk.PhotoImage(img.resize((500,400),Image.ANTIALIAS))
    #image = tk.PhotoImage(file = 'price_data.png')
    imageLabel.configure(image = image)
    imageLabel.image = image
    trade_window.after(1000, display_data)

def connect_to_socket():
    base_url = "https://api.gemini.com/v1"
    response = requests.get(base_url + "/pubticker/"+str(symbol_entry.get())+"usd")
    data = response.json()

    ticker.set(str(data))
    trade_window.after(1000, connect_to_socket)
def close():
   trade_window.quit()
def Cancel_Orders():
    gemini_api_secret = str(keys_dict['gemini_api_secret']).encode()
    base_url = "https://api.gemini.com"
    endpoint = "/v1/order/cancel/all"
    url = base_url + endpoint
    t = datetime.datetime.now()
    payload_nonce = str(int(time.mktime(t.timetuple()) * 1000)) + str(np.random.randint(1000, 10000))
    payload = {
        "nonce": payload_nonce,
    "request": "/v1/order/cancel/all"
    }
    #REALLY NEED TO MAKE SURE WE ARE DOING THE RIGHT THING HERE HAVE A VERIFICATION STEP, LIKE ARE YOU REALLY FUCKING
    #SURE YOU WANT TO YOLO.

    encoded_payload = json.dumps(payload).encode()
    b64 = base64.b64encode(encoded_payload)
    signature = hmac.new(gemini_api_secret, b64, hashlib.sha384).hexdigest()

    request_headers = {'Content-Type': "text/plain",
                       'Content-Length': "0",
                       'X-GEMINI-APIKEY': str(keys_dict['gemini_api_key']),
                       'X-GEMINI-PAYLOAD': b64,
                       'X-GEMINI-SIGNATURE': signature,
                       'Cache-Control': "no-cache"}

    response = requests.post(url,
                             data=None,
                             headers=request_headers)
    new_order = response.json()
    print(new_order)
main_trade_window = tk.Tk()
main_trade_window.attributes('-fullscreen', True)
#sand_window.geometry("1200x1200")
# Make Window into a scrolling frame
container = tk.Frame(main_trade_window)
#canvas = tk.Canvas(container,height=1100,width=1600)
canvas = tk.Canvas(container)
scrollbarh = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
scrollbarv = tk.Scrollbar(container, orient="vertical", command=canvas.yview)

canvas.configure(yscrollcommand=scrollbarv.set)
canvas.configure(xscrollcommand=scrollbarh.set)
scrollbarh.pack(side="bottom", fill="y")
scrollbarv.pack(side="right", fill="y")
container.pack(fill=tk.BOTH, expand=True)
canvas.pack(side="left", fill="both", expand=True)
trade_window = tk.Frame(canvas)
trade_window.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=trade_window, anchor="nw")
key = tk.StringVar('')

timestep = tk.StringVar()
timestep.set("1m")
timestep_entry = tk.OptionMenu(trade_window,timestep,*["1m","5m","15m","30m","1hr","6hr","1day"])
timestep_entry.grid(row=0, column=2)#pack()
#
label_timerange = tk.StringVar()
label_timerange.set("Time Range(number of price data points)")
label_timerange_Dir = tk.Label(trade_window, textvariable=label_timerange)
label_timerange_Dir.grid(row=1, column=0)

timerange_entry = tk.Scale(trade_window, from_= 5, to=1400, orient=tk.HORIZONTAL, length=1000,tickinterval=100)
timerange_entry.set(100)
timerange_entry.grid(row=1, column=1)
#
labelText=tk.StringVar()
labelText.set("Enter Symbol(E.G eth)")
labelDir=tk.Label(trade_window, textvariable=labelText, height=4)
labelDir.grid(row=0, column=0)

# base_url = "https://api.gemini.com/v1"
# response = requests.get(base_url + "/symbols")
# symbols_array = response.json()
# print(symbols_array)

symbol = tk.StringVar()
symbol.set('eth')
symbol.trace('w',Set_Balances)
symbol_entry = tk.Entry(trade_window, textvariable=symbol)
symbol_entry.grid(row=0, column=1)

label_favsymbol = tk.StringVar()
label_favsymbol.set("Favorite Symbols")
label_favsymbol_Dir = tk.Label(trade_window, textvariable=label_favsymbol)
label_favsymbol_Dir.grid(row=1, column=2)
favsymbols = ['btc','eth','sol','usdc','ape','doge']
favsymbol = tk.StringVar()
favsymbol.set(symbol_entry.get())
favsymbols_entry = tk.OptionMenu(trade_window,favsymbol,*favsymbols)
def set_symbol(*args):
    symbol.set(favsymbol.get())
favsymbol.trace('w',set_symbol)
favsymbols_entry.grid(row=2, column = 2)
button_display_prices = tk.Button(trade_window, text="Display Prices",highlightbackground='#3E4149', command=display_data)
button_display_prices.grid(row=2, column=0)

imageLabel = tk.Label(trade_window)
imageLabel.grid(row=2, column=1)


button_display_ticker = tk.Button(trade_window, text="Display Ticker",highlightbackground='#3E4149', command=connect_to_socket)
button_display_ticker.grid(row=3, column=0)

label_ticker = tk.StringVar()
label_ticker.set("Ticker")
label_bids_Dir = tk.Label(trade_window, textvariable=label_ticker)
label_bids_Dir.grid(row=3, column=1)
ticker = tk.StringVar()
ticker_value = tk.Label(trade_window, textvariable=ticker)
ticker_value.grid(row=3, column=2)

user_id = tk.IntVar()

labelText=tk.StringVar()
labelText.set("Enter User Id")
labelDir=tk.Label(trade_window, textvariable=labelText, height=4)
labelDir.grid(row=4, column=1)


id_entry = tk.Entry(trade_window, textvariable=user_id)
id_entry.delete(0, tk.END)
id_entry.insert(tk.END,'123456789')
id_entry.grid(row=4, column=2)

label_key = tk.Label(trade_window, textvariable=key)
label_key.grid(row=4, column=3)

button_find_user = tk.Button(trade_window, text="Find User",highlightbackground='#3E4149', command=Set_User)
button_find_user.grid(row=4, column=0)

label_usdbalance = tk.StringVar()
label_usdbalance.set("USD BALANCE")
label_usdbalance_Dir = tk.Label(trade_window, textvariable=label_usdbalance)
label_usdbalance_Dir.grid(row=5,column = 0)
usdbalance = tk.StringVar()
usdbalance_value = tk.Label(trade_window, textvariable=usdbalance)
usdbalance_value.grid(row=5,column = 1)

label_symbolbalance = tk.StringVar()
label_symbolbalance.set("Symbol Balance")
label_symbolbalance_Dir = tk.Label(trade_window, textvariable=label_symbolbalance)
label_symbolbalance_Dir.grid(row=5,column = 2)
symbolbalance = tk.StringVar()
symbolbalance_value = tk.Label(trade_window, textvariable=symbolbalance)
symbolbalance_value.grid(row=5,column = 3)

labelamount=tk.StringVar()
labelamount.set("Enter amount")
labelamountDir=tk.Label(trade_window, textvariable=labelamount)
labelamountDir.grid(row=6,column = 0)
amount  = tk.StringVar()
amount.trace('w',update_tradetotal)
amount_entry = tk.Entry(trade_window, textvariable=amount)
amount_entry.grid(row=6,column = 1)
#
# label_batch_size_entry = tk.IntVar()
# label_batch_size_entry.set("batch size")
# label_batch_size_entry_Dir = tk.Label(scrollable_frame, textvariable=label_batch_size_entry)
# label_batch_size_entry_Dir.grid(row=4,column = 2)
# batch_size_entry = tk.Entry(scrollable_frame, textvariable='4')
# batch_size_entry.insert(tk.END,'10')
# batch_size_entry.grid(row=4,column = 3)
#
labelprice=tk.StringVar()
labelprice.set("Enter Price")
labelpriceDir=tk.Label(trade_window, textvariable=labelprice, height=4)
labelpriceDir.grid(row=6,column = 2)

price  = tk.StringVar()
price.trace('w',update_tradetotal)
price_entry = tk.Entry(trade_window, textvariable=price)
price_entry.grid(row=6,column = 3)

label_tradetotal = tk.StringVar()
label_tradetotal.set("Total Trade in USD")
label_mean_angle_Dir = tk.Label(trade_window, textvariable=label_tradetotal)
label_mean_angle_Dir.grid(row=7,column = 1)
tradetotal = tk.StringVar()

mean_angle_value = tk.Label(trade_window, textvariable=tradetotal)
mean_angle_value.grid(row=7,column = 2)

button_buy = tk.Button(trade_window, text="Buy",highlightbackground='#0000ff', command = lambda: Gemini_Trade(symbol_entry.get(), float(amount_entry.get()),
                                                                                float(price_entry.get()), 'buy',"exchange limit", keys_dict['gemini_api_key'], keys_dict['gemini_api_secret']))
button_buy.grid(row=7,column = 0)
button_sell = tk.Button(trade_window, text="Sell",highlightbackground='#00ff00', command = lambda: Gemini_Trade(symbol_entry.get(), float(amount_entry.get()),
                                                                                  float(price_entry.get()), 'sell',"exchange limit", keys_dict['gemini_api_key'], keys_dict['gemini_api_secret']))
button_sell.grid(row=7,column = 3)

button_quit = tk.Button(trade_window, text="QUIT", highlightbackground='#ff0000',
                        command=close)
button_cancel = tk.Button(trade_window, text="Cancel Orders", highlightbackground='#ffff00',
                        command=Cancel_Orders)
label_results = tk.StringVar()
label_results.set("Trade Results")
label_results_Dir = tk.Label(trade_window, textvariable=label_results)
label_results_Dir.grid(row=8,column = 0)
results = tk.StringVar()
results_value = tk.Label(trade_window, textvariable=results)
results_value.grid(row=8,column = 1)
button_quit.grid(row=10,column = 0)
button_cancel.grid(row=9,column = 0)
trade_window.mainloop()


