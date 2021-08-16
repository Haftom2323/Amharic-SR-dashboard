from os import write
import sys
sys.path.insert(0, './scripts')
from model_inference import ModelInference
import streamlit as st
import scipy.io.wavfile as wav



def file_uploader():
    uploaded_file = st.file_uploader("Upload Files",type=['wav'])
    return uploaded_file

@st.cache()  
# defining the function which will make 
# the prediction using data about the users 
def prediction(file):   
    sr,y = wav.read(file)
    mi = ModelInference(y)
    result = mi.get_prediction()
    return result

    
def main_page():
    st.markdown('<h2>Amharic Speech To Text</h2>', unsafe_allow_html=True)
    st.write("Upload your audio file")
    file = file_uploader()

    result = ""
    
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Transcribe"): 
        result = prediction(file)
        st.audio(file, format='audio/wav')
        st.write(result)
        # st.success('The user is {}'.format(result))     
  


     
if __name__=='__main__': 
    main_page()