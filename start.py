# Topological Time Series Analysis for Trading in Financial Markets
# Peter Fumich
# Friday 13 May 2022
# # # #
# # # #
# Note: Although the core motivation is Topological Time Series Analysis, we found that a few auxilliary programs were
#       Beneficial. This start script serves as a method for running different files using the tkinter gui library.
import sys
import os
import tkinter

main_window=tkinter.Tk()
# main_window.geometry("500x200")
def Trade():
    os.system('python3 Gemini_Trade.py')
def Sandbox():
    os.system('python3 Sandbox.py')
def close():
    print('Goodbye')
    main_window.quit()


B0 = tkinter.Button(main_window, text="QUIT", highlightbackground='#ff0000',height = 10, width = 50,
                        command=close)
B0.pack()
B1=tkinter.Button(main_window, text="Sandbox",highlightbackground='#ffff00',height = 10, width = 50, command= Sandbox)
B1.pack()
B2=tkinter.Button(main_window, text="Trade",highlightbackground='#00ff00',height = 10, width = 50, command= Trade)
B2.pack()
main_window.mainloop()

