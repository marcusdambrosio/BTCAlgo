import smtplib


def send_email(message, subject = 'No Subject'):
    # create smtp object
    smtpObj = smtplib.SMTP('smtp.gmail.com', 587)

    # Say hello!
    smtpObj.ehlo()

    # start TLS Encryption
    smtpObj.starttls()

    sender = 'marcusdambrosio@gmail.com'
    sender_password = 'placeholder'
    smtpObj.login(sender, sender_password)

    if subject:
        send = smtpObj.sendmail(sender, sender, f'Subject: {subject}\n{message}')


    else:
        send = smtpObj.sendmail(sender, sender, message +'\n\nThis is an automated email.')

    if not send:


        print('EMAIL SENT')
    smtpObj.quit()


