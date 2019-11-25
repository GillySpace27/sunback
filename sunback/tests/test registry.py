import winreg as wr

aReg = wr.ConnectRegistry(None, wr.HKEY_CURRENT_USER)

aKey = wr.OpenKey(aReg, r"Control Panel\Desktop")


for i in range(1024):
    try:
        asubkey_name = wr.EnumValue(aKey, i)
        print(asubkey_name)
        # asubkey = wr.OpenKey(aKey, asubkey_name)
        # print(asubkey)
        # val = wr.QueryValueEx(asubkey, "DisplayName")
        # print(val)
    except EnvironmentError:
        break
