FROM python:3.10
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip install -r requirements.txt
RUN mv _tree.py ../usr/local/lib/python3.10/site-packages/shap/explainers
RUN mv BorutaShap.py ../usr/local/lib/python3.10/site-packages
ENV MONGO_DB_URL="mongodb+srv://user1:eESKWoJHtkCcTaRZ@mentalhealth.qsf7hvl.mongodb.net/?retryWrites=true&w=majority"
CMD streamlit run pipeline_api.py --server.port $PORT