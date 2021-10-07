
# run it with:
# python3 app.py

#import the necessary libraries
from flask import Flask, render_template , request,redirect,send_file
import pickle,os,glob
from flask.wrappers import Request
import threading

import numpy as np
import pandas as pd
from rdkit import rdBase   
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

app = Flask(__name__)
# load the model from disk
#test()

#print(select)

#model = pickle.load(open('finalized_model_96_new.pkl', 'rb'))
#model2 = pickle.load(open('finalized_model_ethanol_98%.pkl', 'rb'))
def getAromaticProportion(m):
    aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aromatic = 0
    for i in aromatic_list:
        if i:
            aromatic += 1
    heavy_atom = Lipinski.HeavyAtomCount(m)
    return aromatic / heavy_atom


def predictSingle(smiles, model):
    """
    This function predicts the four molecular descriptors: the octanol/water partition coefficient (LogP),
    the molecular weight (Mw), the number of rotatable bonds (NRb), and the aromatic proportion (AP) 
    for a single molecule
    
    The input arguments are SMILES molecular structure and the trained model, respectively.
    """
    
    # define the rdkit moleculer object
    mol = Chem.MolFromSmiles(smiles)
    
    # calculate the log octanol/water partition descriptor
    single_MolLogP = Descriptors.MolLogP(mol)
    
    # calculate the molecular weight descriptor
    single_MolWt   = Descriptors.MolWt(mol)
    
    # calculate of the number of rotatable bonds descriptor
    single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
    
    # calculate the aromatic proportion descriptor
    single_AP = getAromaticProportion(mol)

    # Calculate ring count 
    single_RC= Descriptors.RingCount(mol)

    # Calculate TPSA 
    single_TPSA=Descriptors.TPSA(mol)

    # Calculate H Donors  
    single_Hdonors=Lipinski.NumHDonors(mol)

    # Calculate saturated Rings 
    single_SR= Lipinski.NumSaturatedRings(mol) 

    # Calculate Aliphatic rings 
    single_AR =Lipinski.NumAliphaticRings(mol)
    
    # Calculate Hydrogen Acceptors 
    single_HA = Lipinski.NumHAcceptors(mol)

    # Calculate Heteroatoms
    single_Heter = Lipinski.NumHeteroatoms(mol)

    # put the descriptors in a list
    rows = np.array([single_MolLogP, single_MolWt, single_NumRotatableBonds, single_AP,single_RC,single_TPSA,single_Hdonors,single_SR,single_AR,single_HA,single_Heter])
    
    # add the list to a pandas dataframe
    #single_df = pd.DataFrame(single_list).T
    baseData = np.vstack([rows])
    # rename the header columns of the dataframe
    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    #descriptors =np.array(descriptors) 
    #preds=loaded_model.predict(descriptors)
    
    return model.predict(descriptors)

@app.route('/')
def index():
    return render_template(
        'sub.html',
        data=[{'name':'Aquesous'}, {'name':'Ethanol'}, {'name':'Benzene'},{'name':'Acetone'}])

@app.route("/test" , methods=['GET', 'POST'])
def test(): 
    #def handle_view(select):
    global select 
    global model 
    select = request.form.get('comp_select')
    select = str(select)
    if select =='Aquesous':
        model=pickle.load(open('finalized_model_96_new.pkl', 'rb'))
    elif select =='Ethanol':
        model=pickle.load(open('finalized_model_ethanol_98%.pkl', 'rb'))
    elif select == 'Benzene':
        model=pickle.load(open('Finalized_model_Benzene_92%.pkl', 'rb'))
    else:
        model=pickle.load(open('Finalized_model_Acetone_98%.pkl', 'rb'))   

         #return "Thanks"
   #print(model)
    return render_template('sub.html',solvent_text = "The Selected Solvent is {}".format(select)) # just to see what select isprint(test.model)
#threading.thread.start_new_thread(handle_sub_view, select)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        smiles = request.form["smiles"]
        #print(smiles)
    predOUT = predictSingle(smiles, model)
    #predOUT = predOUT +0.20

    return render_template('sub.html', prediction_text = "The log S is {}".format(predOUT))
    #return render_template('sub.html',resu= "The log S is {}".format(predOUT))  
def generate(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        desc_MolLogP = Crippen.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
        desc_AromaticProportion = getAromaticProportion(mol)
        desc_Ringcount        =   Descriptors.RingCount(mol)
        desc_TPSA = Descriptors.TPSA(mol)
        desc_Hdonrs=Lipinski.NumHDonors(mol)
        desc_SaturatedRings = Lipinski.NumSaturatedRings(mol)   
        desc_AliphaticRings = Lipinski.NumAliphaticRings(mol) 
        desc_HAcceptors = Lipinski.NumHAcceptors(mol)
        desc_Heteroatoms = Lipinski.NumHeteroatoms(mol)
        #desc_molMR=Descriptors.MolMR(mol)
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion,desc_Ringcount,desc_TPSA,desc_Hdonrs,desc_SaturatedRings,desc_AliphaticRings,desc_HAcceptors,desc_Heteroatoms])

        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors

app.config["UPLOAD_PATH"]=  'static'
app.config["DOWNLOAD_PATH"]='C:/Users/ali/Desktop/solub_herokuu-main/static/downloads'
@app.route('/upload_file', methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        #solvent= request.form.get('solvent_names')
        dir = app.config["UPLOAD_PATH"]
        for zippath in glob.iglob(os.path.join(dir, '*.csv')):
            os.remove(zippath)
        #os.remove("static/uploads" +item) 
        f=request.files['file_name']
        #print(f)
        #filepath=os.path.join('static',f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
        #f.save(filepath)
        #return render_template("upload_file.html",msg="File has been uploaded")
        data = pd.read_csv(os.path.join(app.config['UPLOAD_PATH'], f.filename))
        #print(data)
        data=data.smiles
        #loaded_model= pickle.load(open('/content/drive/MyDrive/KIT/finalized_model_96_new.pkl', 'rb'))
        descriptors =generate(data)
        descriptors =np.array(descriptors) 
        preds=model.predict(descriptors)
        #print(preds)
        data1=pd.DataFrame(preds, columns=['Predictions']) 
        #data['Predictions'] = preds
        result = pd.concat([data, data1], axis=1)
        filepath=os.path.join('static','out'+'.csv')
        result.to_csv(filepath)
        return send_file(filepath, as_attachment=True)
    return render_template("upload_file.html", msg="Please choose a 'csv' file with smiles")    
if __name__ == "__main__":
    app.run(debug=True, port=7000)

