
def convertTime(string):

    timeAsString = string
    timeAsString = timeAsString.replace('-', ':')
    print('Time string is: ', timeAsString)
    timeSplitted = timeAsString.split(':')

    if len(timeSplitted) == 4:
        print('Split time has 4 entries!', flush=True)
        days = int(timeSplitted[0])
        hours = int(timeSplitted[1])
        minutes = int(timeSplitted[2])
        seconds = int(timeSplitted[3])

    elif len(timeSplitted) == 3:
        print('Split time has 3 entries!', flush=True)
        days = 0
        hours = int(timeSplitted[0])
        minutes = int(timeSplitted[1])
        seconds = int(timeSplitted[2])
    elif len(timeSplitted) == 2:
        print('Split time has 2 entries!', flush=True)
        days = 0
        hours = 0
        minutes = int(timeSplitted[0])
        seconds = int(timeSplitted[1])
    elif len(timeSplitted) == 1:
        print('Split time has 1 entries!', flush=True)
        days = 0
        hours = 0
        minutes = 0
        seconds = int(timeSplitted[0])
    else:
        print('Some error in the time format!', flush=True)
        return -1

    timeInSeconds = seconds + minutes*60 + hours*60*60 + days*24*60*60

    print('Time in seconds ' + str(timeInSeconds))
    return timeInSeconds