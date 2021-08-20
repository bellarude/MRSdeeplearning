import smtplib

def textMe(string):
    # needs to abilitate less secure app: https://www.google.com/settings/security/lesssecureapps
    # NB: avoid ":" in the string!
    content = string
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    address = 'rudy.rizzo.tv@gmail.com'
    mail.login('amsmdeepmrs@gmail.com', 'amsmdeepmrs20')
    mail.sendmail('amsmdeepmrs@gmail.com', address, content)
    mail.close()
    print(">>> sent E-mail @" + address)