FROM continuumio/miniconda3:latest

RUN conda install -y -c anaconda numpy pandas scikit-learn beautifulsoup4 flask gensim && \
	conda install -c conda-forge tqdm && \
    pip install git+https://github.com/kmike/pymorphy2.git && \
    pip install -U flask-cors
 
EXPOSE 10000

WORKDIR /home/user/clones_mini3

CMD ["/bin/bash"]