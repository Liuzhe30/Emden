import time
import uuid
import os

from celery import Celery
from dateutil import parser
from flask import Flask, render_template
from flask import request, url_for, redirect
from flask_mail import Mail, Message
import numpy as np
from Bio import ExPASy
from Bio import SwissProt

# from online_predict.MiPPIs.main import MiPPIs
# from online_predict.MiPPIs.main import MiPPIs
# celery -A app:celery worker -l info
# celery -A app:celery worker -l info -P gevent

from utils.dbutils import DBConnection
import os

residue_dict = {
'GLY': 'G', 'ALA': 'A', 'LEU': 'L', 'ILE': 'I', 'VAL': 'V', 
'PRO': 'P', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W', 'SER': 'S',
'THR': 'T', 'CYS': 'C', 'MET': 'M', 'ASN': 'N', 'GLN': 'Q',
'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'HIS': 'H'
}

def app_cfg_init():
    flask_app = Flask(__name__)
    flask_app.secret_key = 'emden_secret'
    flask_app.debug = True
    return flask_app


def email_cfg_init(app):
    # Flask-Mail configuration
    app.config['MAIL_SERVER'] = 'smtp.yeah.net'
    app.config['MAIL_PORT'] = 25
    app.config['MAIL_USE_SSL'] = False
    app.config['MAIL_USE_TLS'] = False
    app.config['MAIL_USERNAME'] = 'baoyihangauto@yeah.net'
    app.config['MAIL_PASSWORD'] = 'baoyihangAUTO123'
    app.config['MAIL_DEFAULT_SENDER'] = 'Emden admin'
    app.config['MAIL_ASCII_ATTACHMENTS'] = True
    app.config['MAIL_DEBUG'] = True

    ml = Mail(app)
    return ml


def celery_cfg_init(app):
    
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6381/0'
    print(app.name)
    celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])

    celery.conf.update(app.config)
    CELERY_ACCEPT_CONTENT = ['pickle', 'json']
    return celery


app = app_cfg_init()
celery = celery_cfg_init(app)
mail = email_cfg_init(app)

@celery.task(bind=True)
def predict_async_task(self, input_dic, task_id, email=None, dbc=None):
    """Background task to predict and/or send an email with Flask-Mail."""
    print('get')
    if not dbc:
        dbc = DBConnection()
    try:
        # self.update_state(state="PROGRESS")
        os.system('sh /data/emden/Emden-web/RunSingleRound.sh %s %s %s %s %s' % (input_dic['seq'], task_id, input_dic['mupos'], input_dic['muafter'], input_dic['drugname']))
        predict_result = np.load('/data/emden/Emden-web/model/pred_results/ori/pred_prob' + str(task_id) + '.npy')
        predict_result = float(predict_result[0])
        if email:
            email_data = {
                'subject': 'Predict result from Emden, job id: %s' % task_id,
                'to': email,
                'body': predict_result
            }
            msg = Message(email_data['subject'],
                          sender='guosijia007@yeah.net',
                          recipients=[email_data['to'], 'guosijia007@yeah.net'
                                      ])
            msg.body = email_data['body']
            with app.app_context():
                mail.send(msg)

        dbc.col.update_one({"job_id": task_id},
                           {"$set": {"job_state": "Success", "time": parser.parse(
                               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                     'result': predict_result
                                     }})
    except Exception as exc:
        print(exc)
        print('error')
        dbc.col.update_one({"job_id": task_id},
                           {"$set": {
                               "job_state": "Failed, some errors may have occurred when running the prediction script.",
                               "time": parser.parse(
                                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))}})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/seq_pred", methods=['GET', 'POST'])
def get_seq_pred():
    if request.method == 'GET':
        return render_template('seq_pred.html')

    task_id = str(uuid.uuid4())
    now = parser.parse(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



    input_dict = {}
    input_dict['seq'] = request.form.get('seq').strip()
    if input_dict['seq'][-2].isdigit():
        with ExPASy.get_sprot_raw(input_dict['seq'].strip()) as handle:
            record = SwissProt.read(handle)
        input_dict['seq'] = record.sequence
    mu_info = request.form.get('muinfo').strip()
    input_dict['mupos'] = mu_info.strip()[1:-1]
    input_dict['muafter'] = mu_info.strip()[-1]
    input_dict['drugname'] = request.form.get('drugname').strip()

    # Register the task in mongoDB, the task status is started
    job = {'job_id': task_id, 'job_name': '', 'job_state': 'start', 'result': [], 'time': now,
           'pdb_id': '', 'mutated_id': '', 'partner_id': '', 'mutated_pos':'',
           'seq': input_dict['seq'], 'mupos': input_dict['mupos'], 'muafter': input_dict['muafter'], 'drugname': input_dict['drugname'],
           }
    dbc = DBConnection()
    dbc.col.insert_one(job)


    # email = request.form.get('form1_email')
    email = ''

    # Update task results and status to MongoDB running
    if not 'seq' in input_dict or not 'mupos' in input_dict or not 'muafter' in input_dict or not 'drugname' in input_dict:
        dbc.col.update_one({"job_id": task_id},
                           {"$set": {"job_state": "Error, illegal input was detected !", "time": parser.parse(
                               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))}})
        return redirect('/Emden/job/{}'.format(task_id))
    else:
        dbc.col.update_one({"job_id": task_id},
                           {"$set": {"job_state": "Running", "time": parser.parse(
                               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))}})

        # Asynchronous call, predict and/or send mail
        # TODO
        predict_async_task.delay(input_dict, task_id, email)

        # Jump to the task result page
        return redirect('/Emden/job/{}'.format(task_id))
        # return redirect(url_for('get_result', job_id=task_id))

    return redirect(url_for('index'))

@app.route("/job/<job_id>")
def get_job(job_id):
    dbc = DBConnection()
    results = dbc.col.find_one({"job_id": job_id}, {'_id': False})
    return render_template('job.html', results=results)

@app.route("/job_3d/<job_id>")
def get_job_3d(job_id):
    dbc = DBConnection()
    results = dbc.col.find_one({"job_id": job_id}, {'_id': False})
    return render_template('job_3d.html', results=results)

@app.route("/download")
def download():
    return render_template('download.html')

@app.route("/contact")
def contact():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)

