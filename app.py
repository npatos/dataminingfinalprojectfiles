from flask import Flask,render_template,request 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel    
from flask import Flask, render_template, request, jsonify

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel      
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('model.h5')
app = Flask(__name__)



def load_data(data):
    df= pd.read_csv(data, sep= ';', error_bad_lines= False, encoding= 'latin-1')
    df=df.head(500)
    return df
df=load_data('https://raw.githubusercontent.com/tttgm/fellowshipai/master/book_crossing_dataset/BX-Books.csv')

@app.route('/')
def login_page():
  df=pd.read_csv('https://raw.githubusercontent.com/npatos/dataminingfinalprojectdata/main/books_summary_noerros.csv')
  titles = df['book_title']
  authors=df['book_author']
  years=df['year_of_publication']
  images = df['image_url_l']
  df_rating=pd.read_csv('https://raw.githubusercontent.com/npatos/dataminingfinalprojectdata/main/books_summary_noerros.csv')
  titles_rating = df_rating['book_title']
  authors_rating=df_rating['book_author']
  years_rating=df_rating['year_of_publication']
  images_rating = df_rating['image_url_l']
  # COLLABORATIVE
  df= pd.read_csv('https://raw.githubusercontent.com/npatos/dataminingfinalprojectdata/main/Finalcollab.csv')
  coll_titles = df['book_title']
  coll_authors=df['book_author']
  coll_years=df['year_of_publication']
  coll_images= df['image_url_l']
  return render_template('index.html',coll_images=coll_images,coll_years=coll_years,coll_titles=coll_titles,coll_authors=coll_authors,title = titles,author=authors,year = years,image=images,titles_rating=titles_rating,authors_rating=authors_rating,years_rating=years_rating,images_rating=images_rating)

@app.route('/book/<coll_titles>', methods=['GET'])
def book_collaborative_recommend(coll_titles):
  name = str(coll_titles)
  combine_book_rating_data=pd.read_csv('https://raw.githubusercontent.com/npatos/dataminingfinalprojectdata/main/Finalcollab.csv')
  books_df_s=combine_book_rating_data[combine_book_rating_data['book_title']==name]
  titles_searched = books_df_s['book_title']
  authors_searched=books_df_s['book_author']
  year_searched = books_df_s['year_of_publication']
  images_searched = books_df_s['image_url_l']
  user_id = books_df_s['user']
  user_id=user_id.iloc[0]
  user_r = user_id
  b_id =list(combine_book_rating_data.user.unique())
  book_arr = np.array(b_id) #get all book IDs
  user = np.array([user_r for i in range(len(b_id))])
  pred = model.predict([book_arr, user])
  pred = pred.reshape(-1) #reshape to single dimension
  pred_ids = (-pred).argsort()[0:10]
  top10 = combine_book_rating_data.iloc[pred_ids]
  f=['book_title','book_author','year_of_publication','image_url_l']
  displ=(top10[f])
  c_title = displ['book_title']
  c_authors = displ['book_author']
  c_small_image_url= displ['image_url_l']
  c_years= displ['year_of_publication']
  return render_template('recommendedbooks.html',year_searched=year_searched,c_years=c_years,images_searched=images_searched,authors_searched=authors_searched,titles_searched=titles_searched,c_title=c_title,c_authors=c_authors,c_small_image_url=c_small_image_url)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
