from fastapi import FastAPI  # Importo las librerias que utilizare
import pandas as pd
import uvicorn
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Lectura de todos los CSV
developer = pd.read_csv('Funciones/funcion1_developer.csv', low_memory=False)
user_data = pd.read_csv('Funciones/Funcion2_userdata.csv', low_memory=False)
UsersGenre= pd.read_csv('Funciones/UsersGenre_funcion3.csv', low_memory=False)
best_developer_year= pd.read_csv('Funciones/BestDeveloper_funcion4.csv', low_memory=False)
developer_reviews_analysis = pd.read_csv('Funciones/sentiment_analysis_funcion.csv', low_memory=False)
render_model= pd.read_csv('modeloo_render.csv',low_memory=False)   

# Convertir a archivos Parquet
developer.to_parquet('Funciones/funcion1_developer.parquet')
user_data.to_parquet('Funciones/Funcion2_userdata.parquet')
UsersGenre.to_parquet('Funciones/UsersGenre_funcion3.parquet')
best_developer_year.to_parquet('Funciones/BestDeveloper_funcion4.parquet')
developer_reviews_analysis.to_parquet('Funciones/sentiment_analysis_funcion.parquet')
render_model.to_parquet('modeloo_render.parquet')


# Funcion def developer

@app.get("/developer/{desarrollador}")
def developer_info(desarrollador: str):
    try:
        # Filtra el DataFrame 'developer' por el nombre del desarrollador
        df_filtered = developer[developer['developer'] == desarrollador].copy()
        
        # Convierte la columna 'release_date' en tipo datetime y extrae el año
        df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'])
        df_filtered['Year'] = df_filtered['release_date'].dt.year
        
        # Calcula el número total de juegos por año
        total_items = df_filtered.groupby('Year')['item_id'].count()
        
        # Calcula el número de juegos gratuitos por año
        free_to_play_items = df_filtered[df_filtered['price'] == 'Free To Play'].groupby('Year')['item_id'].count()
        
        # Calcula el porcentaje de juegos gratuitos por año
        free_to_play_percentage = (free_to_play_items / total_items * 100).fillna(0).astype(int)
        
        # Crea un DataFrame con los resultados
        result = pd.DataFrame({
            'Year': free_to_play_percentage.index,
            'Cantidad_de_Items': total_items.values,
            'Contenido_Free': free_to_play_percentage.values
        })
        
        # Ordena los resultados por año de forma descendente
        result.sort_values('Year', ascending=False, inplace=True)
        
        # Devuelve los resultados como un diccionario de registros
        return result.to_dict(orient="records")
    except Exception as e:
        # En caso de error, devuelve un mensaje explicativo
        return {"error": str(e)}

    
   
if __name__=="__main__":
    uvicorn.run("main:app",port=8000,reload=True) 
    
    
# Funcion def userdata



user_data = pd.read_csv('Funciones/Funcion2_userdata.csv', low_memory=False)

def userdata(User_id: str):
    try:
        # Convertir valores no numéricos en la columna 'price' a NaN
        user_data['price'] = pd.to_numeric(user_data['price'], errors='coerce')

        # Filtrar los datos para el usuario dado
        user_subset = user_data[user_data['user_id'] == User_id]

        # Calcular cantidad de dinero gastado
        dinero_gastado = user_subset['price'].sum()

        # Calcular porcentaje de recomendación
        recomendacion_count = user_subset['recommend'].sum()
        total_reviews = user_subset.shape[0]
        porcentaje_recomendacion = (recomendacion_count / total_reviews) * 100

        # Calcular cantidad de items
        cantidad_items = user_subset.shape[0]

        # Crear el diccionario de retorno
        retorno = {
            "Usuario": User_id,
            "Dinero gastado": f"${dinero_gastado:.2f}",
            "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
            "Cantidad de items": cantidad_items
        }

        return retorno
    except Exception as e:
        # En caso de error, devuelve un mensaje explicativo
        return {"error": str(e)}

@app.get("/userdata/{User_id}")
def user_data_endpoint(User_id: str):
    return userdata(User_id)
    

    
if __name__=="__main__":
    uvicorn.run("main:app",port=8000,reload=True)

# Funcion def UserGenre

UsersGenre = pd.read_csv('Funciones/UsersGenre_funcion3.csv', low_memory=False)
# Convertir la columna 'release_date' a tipo datetime
UsersGenre['release_date'] = pd.to_datetime(UsersGenre['release_date'], errors='coerce')

@app.get("/genero")
def UserForGenre(genero: str):
    try:
        # Filtrar el DataFrame por el género específico
        genre_subset = UsersGenre[UsersGenre['genres'].str.contains(genero, na=False)]

        # Verificar si se encontraron datos para el género específico
        if genre_subset.empty:
            return {"error": f"No se encontraron datos para el género '{genero}'."}

        # Convertir la columna 'release_date' a tipo datetime
        genre_subset['release_date'] = pd.to_datetime(genre_subset['release_date'], errors='coerce')

        # Obtener el usuario con más horas jugadas para el género
        max_playtime_user = genre_subset.loc[genre_subset['playtime_forever'].idxmax()]

        # Calcular las horas jugadas por año para el género
        playtime_by_year = genre_subset.groupby(genre_subset['release_date'].dt.year)['playtime_forever'].sum().reset_index()
        playtime_by_year.columns = ['Año', 'Horas']
        horas_por_año = playtime_by_year.to_dict(orient='records')

        retorno = {
            "Usuario con más horas jugadas para {}".format(genero): max_playtime_user['user_id'],
            "Horas jugadas": horas_por_año
        }

        return retorno
    except Exception as e:
        # En caso de error, devuelve un mensaje explicativo
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
    
    
# Funcion def best_developer_year

# Cargar los datos del archivo CSV 'BestDeveloper_funcion4.csv'
bestdeveloper = pd.read_csv('Funciones/BestDeveloper_funcion4.csv', low_memory=False)
bestdeveloper['release_date'] = pd.to_datetime(bestdeveloper['release_date'], errors='coerce')

@app.get("/best_developer_year")
def best_developer_year(año: int):
    try:
        # Filtrar el DataFrame por el año dado
        year_subset = bestdeveloper[bestdeveloper['release_date'].dt.year == año]

        # Filtrar los juegos recomendados por usuarios (reviews.recommend = True)
        year_subset = year_subset[year_subset['recommend'] == True]

        # Agrupar por desarrollador y contar la cantidad de juegos recomendados
        developer_recommendations = year_subset.groupby('developer').size().reset_index(name='recommendations')

        # Ordenar los desarrolladores por la cantidad de juegos recomendados de forma descendente
        developer_recommendations = developer_recommendations.sort_values(by='recommendations', ascending=False)

        # Tomar los primeros 3 desarrolladores
        top_developers = developer_recommendations.head(3)

        # Crear la lista de resultados
        resultados = [{"Puesto {}: {}".format(idx+1, row['developer']): row['recommendations']} for idx, row in top_developers.iterrows()]

        return resultados
    except Exception as e:
        # En caso de error, devuelve un mensaje explicativo
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
    
    
# Función de developer_reviews_analysis 

# Cargar los datos del archivo CSV 'sentiment_analysis_funcion.csv'
developer_reviews_analysis = pd.read_csv('Funciones/sentiment_analysis_funcion.csv', low_memory=False)

@app.get("/desarrolladora")
def developer_reviews_analysis_endpoint(desarrolladora: str):
    try:
        # Filtrar el DataFrame por la desarrolladora dada
        data_desarrolladora = developer_reviews_analysis[developer_reviews_analysis['developer'] == desarrolladora]

        # Contar las reseñas positivas y negativas
        positivos = data_desarrolladora[data_desarrolladora['sentiment_analysis'] == 1.0].shape[0]
        negativos = data_desarrolladora[data_desarrolladora['sentiment_analysis'] == 0.0].shape[0]

        # Crear el diccionario de retorno
        retorno = {desarrolladora: {'Positive': positivos, 'Negative': negativos}}

        return retorno
    except Exception as e:
        # En caso de error, devuelve un mensaje explicativo
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
# Función Sistema de recomendación Usuario-Item


@app.get("/juegos_usuario_item/{user_id}")

def recomendacion_usuario(user_id: str):
    # Encuentra con el user_id los juegos recomendados
    if user_id in render_model['user_id'].values:
        juegos = render_model.index[render_model['user_id'] == user_id].tolist()[0]
        
        juego_caracteristicas = render_model.iloc[juegos, 3:].values.reshape(1, -1)
        
        render_similitud = cosine_similarity(render_model.iloc[:, 3:], juego_caracteristicas)
        juegos_similaresrecomend = render_similitud.argsort(axis=0)[::-1][1:6]
        juegos_similaresrecomend = juegos_similaresrecomend.flatten()[1:]
        juegos_similares = render_model.iloc[juegos_similaresrecomend]['app_name']
        
        return juegos_similares  
    else:
        return "El juego con el user_id especificado no existe en la base de datos."


# Ejecutar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)