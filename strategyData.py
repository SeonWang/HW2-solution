# SimpleHistoryExample.py
from __future__ import print_function
from __future__ import absolute_import

from optparse import OptionParser

import requests
from bs4 import BeautifulSoup
import lxml
import os
import platform as plat
import sys
# if sys.version_info >= (3, 8) and plat.system().lower() == "windows":
#     # pylint: disable=no-member
#     with os.add_dll_directory(os.getenv('BLPAPI_LIBDIR')):
#         import blpapi
# else:
import blpapi
import pandas as pd
import numpy as np
from datetime import *

def parseCmdLine():
    parser = OptionParser(description="Retrieve reference data.")
    parser.add_option("-a",
                      "--ip",
                      dest="host",
                      help="server name or IP (default: %default)",
                      metavar="ipAddress",
                      default="localhost")
    parser.add_option("-p",
                      dest="port",
                      type="int",
                      help="server port (default: %default)",
                      metavar="tcpPort",
                      default=8194)

    (options, args) = parser.parse_args()

    return options

def component_fun(bbg_identifier, startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        for stock in bbg_identifier:
            request.getElement("securities").appendValue(stock)
        request.getElement("fields").appendValue("PX_VOLUME")
        request.getElement("fields").appendValue("BS_SH_OUT")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        try:
                            histdata.append([fd.getElementAsString("date"),
                                             fd.getElementAsFloat("PX_VOLUME"),
                                             fd.getElementAsFloat("BS_SH_OUT") * 1000000])
                        except blpapi.exception.NotFoundException:
                            pass
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def component_cap_rank_fun(bbg_identifier, startDate):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        for stock in bbg_identifier:
            request.getElement("securities").appendValue(stock)
        request.getElement("fields").appendValue("CUR_MKT_CAP")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "YEARLY")
        request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", startDate)
        request.set("maxDataPoints", 100)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    name = msg.getElement("securityData").getElementAsString("security")
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        try:
                            histdata.append([name, fd.getElementAsString("date"),
                                             fd.getElementAsFloat("CUR_MKT_CAP")])
                        except blpapi.exception.NotFoundException:
                            pass
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def spx_roe(startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue("SPX Index")
        request.getElement("fields").appendValue("RETURN_COM_EQY")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        try:
                            histdata.append([fd.getElementAsString("date"),
                                             fd.getElementAsFloat("RETURN_COM_EQY")])
                        except blpapi.exception.NotFoundException:
                            pass
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def bond_yield(bond, startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue(bond)
        request.getElement("fields").appendValue("PX_MID")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        try:
                            histdata.append([fd.getElementAsString("date"),
                                             fd.getElementAsFloat("PX_MID")])
                        except blpapi.exception.NotFoundException:
                            pass
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def ivv_fun(startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue("IVV US Equity")
        request.getElement("fields").appendValue("OPEN")
        request.getElement("fields").appendValue("HIGH")
        request.getElement("fields").appendValue("LOW")
        request.getElement("fields").appendValue("PX_OFFICIAL_CLOSE")
        request.getElement("fields").appendValue("PX_CLOSE_1D")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        try:
                            histdata.append([fd.getElementAsString("date"),
                                             fd.getElementAsFloat("OPEN"),
                                             fd.getElementAsFloat("HIGH"),
                                             fd.getElementAsFloat("LOW"),
                                             fd.getElementAsFloat("PX_OFFICIAL_CLOSE"),
                                             fd.getElementAsFloat("PX_CLOSE_1D")])
                        except blpapi.exception.NotFoundException:
                            pass
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def brent_fun(startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue("CO1 Comdty")
        request.getElement("fields").appendValue("CHG_PCT_1D")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        histdata.append([fd.getElementAsString("date"),
                                         fd.getElementAsFloat("CHG_PCT_1D")])
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def usd_fun(startDate, endDate, maxPoints):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")

        request.getElement("securities").appendValue("DXY Curncy")
        request.getElement("fields").appendValue("CHG_PCT_1D")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        # request.set("overrideOption", "OVERRIDE_OPTION_GPA")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", maxPoints)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)
        # Process received events
        list = []
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            histdata = []
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":
                    for fd in msg.getElement("securityData").getElement("fieldData").values():
                        histdata.append([fd.getElementAsString("date"),
                                         fd.getElementAsFloat("CHG_PCT_1D")])
                    list.extend(histdata)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completly received, so we could exit
                return list
    finally:
        # Stop the session
        session.stop()

def spx_components():
    # URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # gspc_page = requests.get(URL)
    # soup = BeautifulSoup(gspc_page.content, 'html.parser')
    # table_html = soup.findAll('table', {'id': 'constituents'})
    # df = pd.read_html(str(table_html))[0]
    df = pd.read_csv('constituents.csv')
    component = df['Symbol'].apply(lambda x: x + ' US Equity')
    return list(component)

def histdata_adjust(df):
    df['ivv_return'] = (df['CLOSE'] - df['CLOSE_1D']) / df['CLOSE_1D']
    df['inflation'] = df['T_NOTE_YLD'] - df['TIPS_YLD']
    arr = np.array(df['ivv_return'])
    arr[arr > 0] = 1
    arr[arr < 0] = -1
    df['ivv_signal'] = arr
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df

def req_hisdata(startDate, endDate, train_window, maxPoints):
    print("SimpleHistoryExample")
    try:
        # start = "20200401"
        # end = "20210404"
        # max_points = 150
        start = (startDate + timedelta(days= - 2 * train_window)).strftime('%Y%m%d')
        end = endDate.strftime('%Y%m%d')
        max_points = maxPoints

        # get the turnover ratio
        component_stocks = spx_components()
        slected_stocks = pd.DataFrame(component_cap_rank_fun(component_stocks, start), columns=["name", "date", "cap"])
        slected_stocks = slected_stocks.sort_values(by='cap', ascending=False, ignore_index=True)
        slected_stocks = list(slected_stocks.loc[[0, 49, 99, 149, 199, 249, 299, 349, 399, 449],['name']].iloc[:, 0])

        stocks_hd = pd.DataFrame(component_fun(slected_stocks, start, end, max_points),
                                 columns=["DATE", "PX_VOLUME", "BS_SH_OUT"])
        stocks_hd = stocks_hd[stocks_hd['PX_VOLUME'] > 0]
        stocks_hd.to_csv("stocks_hd.csv")
        stocks_hd['TR'] = stocks_hd['PX_VOLUME'] / stocks_hd['BS_SH_OUT']
        stocks_hd = stocks_hd.groupby(stocks_hd['DATE'], as_index=False).mean()
        tr_hd = stocks_hd[['DATE', 'TR']].set_index('DATE')
        tr_hd.to_csv("tr_hd.csv")

        spx_roe_hd = pd.DataFrame(spx_roe(start, end, max_points), columns=["DATE", "ROE"]).set_index('DATE')
        spx_roe_hd.to_csv("spx_roe_hd.csv")

        t_note_hd = pd.DataFrame(bond_yield("USGG10YR Index", start, end, max_points), columns=["DATE", "T_NOTE_YLD"]).set_index('DATE')
        tips_hd = pd.DataFrame(bond_yield("H15X10YR Index", start, end, max_points), columns=["DATE", "TIPS_YLD"]).set_index('DATE')
        t_note_hd.to_csv('t_note_hd.csv')
        tips_hd.to_csv('tips_hd.csv')

        # brent_hd = pd.DataFrame(brent_fun(start, end, max_points), columns=["DATE", "BRENT_CHG_PCT_1D"]).set_index('DATE')
        # brent_hd.to_csv("brent_hd.csv")
        #
        # usd_hd = pd.DataFrame(usd_fun(start, end, max_points), columns=["DATE", "USD_CHG_PCT_1D"]).set_index('DATE')
        # usd_hd.to_csv("usd.csv")

        ivv_hd = pd.DataFrame(ivv_fun(start, end, max_points), columns=["DATE", "OPEN", 'HIGH', 'LOW', "CLOSE", "CLOSE_1D"]).set_index('DATE')
        ivv_hd.to_csv("ivv_hd.csv")

        df = pd.concat([ivv_hd, tr_hd, spx_roe_hd, t_note_hd, tips_hd], join='inner', axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = histdata_adjust(df)
        df.to_csv('histdata.csv')

        return df

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")