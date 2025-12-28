import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import concurrent.futures
import asyncio
import httpx
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Анализ температурных данных", layout="wide")
st.title("Анализ температурных данных и мониторинг текущей погоды")


@st.cache_data
def generate_initial_data():
    
    # Реальные средние температуры (примерные данные) для городов по сезонам
    seasonal_temperatures = {
        "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
        "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
        "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
        "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
        "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
        "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
        "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
        "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
        "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
        "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
        "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
        "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
        "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
        "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
        "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
    }

    # Сопоставление месяцев с сезонами
    month_to_season = {12: "winter", 1: "winter", 2: "winter",
                       3: "spring", 4: "spring", 5: "spring",
                       6: "summer", 7: "summer", 8: "summer",
                       9: "autumn", 10: "autumn", 11: "autumn"}

    # Генерация данных о температуре
    def generate_realistic_temperature_data(cities, num_years=10):
        dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
        data = []

        for city in cities:
            for date in dates:
                season = month_to_season[date.month]
                mean_temp = seasonal_temperatures[city][season]
                # Добавляем случайное отклонение
                temperature = np.random.normal(loc=mean_temp, scale=5)
                data.append({"city": city, "timestamp": date, "temperature": temperature})

        df = pd.DataFrame(data)
        df["season"] = df["timestamp"].dt.month.map(lambda x: month_to_season[x])
        return df

    # Генерация данных
    data = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))
    data.to_csv("temperature_data.csv", index=False)
    return data


# файл данных генерируется при запуске
initial_data = generate_initial_data()
st.sidebar.write("Файл temperature_data.csv сгенерирован")

# состояния
if "data" not in st.session_state:
    st.session_state.data = None
if "city" not in st.session_state:
    st.session_state.city = None
if "results" not in st.session_state:
    st.session_state.results = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "weather" not in st.session_state:
    st.session_state.weather = None
if "stat_df" not in st.session_state:
    st.session_state.stat_df = None
if "current_temp" not in st.session_state:
    st.session_state.current_temp = None
if "num_rows_head" not in st.session_state:
    st.session_state.num_rows_head = 10

# загрузка
st.sidebar.header("Загрузка данных")

uploaded_file = st.sidebar.file_uploader("Выберите CSV-файл с температурными данными", type=["csv"])

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.sidebar.info(f"Загружен файл: {uploaded_file.name}")
else:
    st.session_state.data = initial_data
    st.sidebar.info("Используются сгенерированные данные temperature_data.csv")


# выбор города из выпадающего списка
if st.session_state.data is not None:
    st.header("Превью данных")
    st.dataframe(st.session_state.data.head())
    
    st.sidebar.header("Выбор города")
    cities = st.session_state.data["city"].unique()
    st.session_state.city = st.sidebar.selectbox("Выберите город для анализа", options=cities)
    
    st.write(f"Выбран город: {st.session_state.city}")


# API-ключ
st.sidebar.header("Текущая погода (OpenWeatherMap)")
api_key = st.sidebar.text_input("Введите API-ключ OpenWeatherMap", type="password", help="Получите бесплатный ключ на openweathermap.org")
if api_key:
    st.session_state.api_key = api_key


# анализ данных
def analyze_data(df):
    start_time = time.time()
    
    result_df = df.sort_values("timestamp").copy()
    
    # скользящее среднее
    result_df["rolling_mean"] = result_df["temperature"].rolling(window=30, min_periods=1).mean()
    
    # по сезонам для каждого города
    stat_df = df.groupby(["city", "season"]).agg(mean_temp=("temperature", "mean"), std_temp=("temperature", "std")).reset_index()
    
    stat_df["upper_bound"] = stat_df["mean_temp"] + 2 * stat_df["std_temp"]
    stat_df["lower_bound"] = stat_df["mean_temp"] - 2 * stat_df["std_temp"]
    
    merged_df = result_df.merge(stat_df[["city", "season", "upper_bound", "lower_bound"]], on=["city", "season"], how="left")
    
    # аномалии
    merged_df["is_anomaly"] = (merged_df["temperature"] > merged_df["upper_bound"]) | (merged_df["temperature"] < merged_df["lower_bound"])
    
    result_df = merged_df.drop(columns=["upper_bound", "lower_bound"])
    result_df['year'] = result_df['timestamp'].dt.year

    time_taken = time.time() - start_time

    return result_df, stat_df, time_taken


# анализ для одного города
def analyze_city_data(city_df):
    city_result = city_df.sort_values("timestamp").copy()
    
    city_result["rolling_mean"] = city_result["temperature"].rolling(window=30, min_periods=1).mean()
    
    city_stat = city_df.groupby("season").agg(mean_temp=("temperature", "mean"), std_temp=("temperature", "std")).reset_index()
    city_stat["city"] = city_df["city"].iloc[0]  
    
    city_stat["upper_bound"] = city_stat["mean_temp"] + 2 * city_stat["std_temp"]
    city_stat["lower_bound"] = city_stat["mean_temp"] - 2 * city_stat["std_temp"]
    
    merged_city = city_result.merge(city_stat[["season", "upper_bound", "lower_bound"]], on="season", how="left")
    
    merged_city["is_anomaly"] = (merged_city["temperature"] > merged_city["upper_bound"]) | (merged_city["temperature"] < merged_city["lower_bound"])
    
    city_result = merged_city.drop(columns=["upper_bound", "lower_bound"])
    
    return city_result, city_stat


# анализ с паралельностью
def analyze_data_parallel(df):
    start_time = time.time()
    
    # по городам
    cities = df["city"].unique()
    city_dfs = [df[df["city"] == city] for city in cities]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_city_data, city_dfs))
    
    all_city_results = []
    all_city_stats = []
    
    for city_result, city_stat in results:
        all_city_results.append(city_result)
        all_city_stats.append(city_stat)
    
    result_df = pd.concat(all_city_results).sort_values("timestamp")
    stat_df = pd.concat(all_city_stats).reset_index(drop=True)
    result_df['year'] = result_df['timestamp'].dt.year

    time_taken = time.time() - start_time

    return result_df, stat_df, time_taken



st.header("Результаты анализа")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Анализ")
    if st.button("Запустить анализ", key="btn_normal"):
        result_df, stat_df, time_taken = analyze_data(st.session_state.data)
        st.session_state.results = result_df
        st.session_state.stat_df = stat_df
        st.session_state.analysis_time = time_taken
        
        st.write(f"Время: {time_taken:.3f} секунд")
        st.write(f"Всего аномалий: {result_df['is_anomaly'].sum()}")
        st.write(f"Городов: {len(stat_df['city'].unique())}")

with col2:
    st.subheader("Параллельный анализ")
    if st.button("Запустить параллельный анализ", key="btn_parallel"):
        result_df, stat_df, time_taken = analyze_data_parallel(st.session_state.data)
        st.session_state.results = result_df
        st.session_state.stat_df = stat_df
        st.session_state.analysis_time_parallel = time_taken
        
        st.write(f"Время: {time_taken:.3f} секунд")
        st.write(f"Всего аномалий: {result_df['is_anomaly'].sum()}")
        st.write(f"Городов: {len(stat_df['city'].unique())}")

if st.session_state.stat_df is not None:
    st.session_state.num_rows_head = st.slider("Сколько строк полученного датасета со статистикой показать", min_value=1, max_value=len(st.session_state.stat_df), value=10, step=1, key="row_head_slider_2")
    st.dataframe(st.session_state.stat_df.head(st.session_state.num_rows_head))

# cравнение
if hasattr(st.session_state, "analysis_time") and hasattr(st.session_state, "analysis_time_parallel"):
    st.subheader("Сравнение скорости")
    diff = abs(st.session_state.analysis_time - st.session_state.analysis_time_parallel)
    st.write(f"Обычный: {st.session_state.analysis_time:.3f} секунд")
    st.write(f"Параллельный: {st.session_state.analysis_time_parallel:.3f} секунд")
    if st.session_state.analysis_time_parallel < st.session_state.analysis_time:
        st.write(f"Параллельный метод быстрее на {diff:.3f} сек")
    else:
        st.write(f"Обычный метод быстрее на {diff:.3f} сек")


# cтатистика по выбранному городу
if st.session_state.city is not None:
    st.header(f"Историческая статистика для города: {st.session_state.city}")

    if st.session_state.stat_df is not None:

        city_stat = st.session_state.stat_df[st.session_state.stat_df["city"] == st.session_state.city]
        
        if not city_stat.empty:
            st.dataframe(city_stat)
            
            if st.session_state.results is not None:
                city_results = st.session_state.results[st.session_state.results["city"] == st.session_state.city]
                st.write(f"Аномалий в этом городе: {city_results['is_anomaly'].sum()}")
        else:
            st.write("Нет статистики для этого города")
    else:
        st.write("Сначала запустите анализ данных")
else:
    st.write("Выберите город")



# визуализация
st.header("Визуализация всего датасета")

tab_all1, tab_all2, tab_all3 = st.tabs(["Общая статистика", "Временные ряды", "Сезонные профили"])

with tab_all1:
    st.subheader("Общая статистика по всем городам")
    
    if st.session_state.data is not None:
        desc_stats = st.session_state.data.groupby("city")["temperature"].describe()
        st.write("Описательная статистика по городам:")
        st.dataframe(desc_stats)
        
        fig_hist_all = px.histogram(
            st.session_state.data,
            x="temperature",
            title="Распределение температур",
            nbins=30)
        st.plotly_chart(fig_hist_all, use_container_width=True)
        
        city_means = st.session_state.data.groupby("city")["temperature"].mean().sort_values()
        fig_bar_all = px.bar(
            x=city_means.index,
            y=city_means.values,
            title="Средние температуры по городам",
            labels={"x": "Город", "y": "Средняя температура"})
        st.plotly_chart(fig_bar_all, use_container_width=True)

with tab_all2:
    st.subheader("Временные ряды всех городов")
    
    if st.session_state.data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            all_cities = st.session_state.data["city"].unique()
            selected_cities = st.multiselect("Выберите города", options=all_cities, default=all_cities[:3])

            show_anomalies = st.checkbox("Показать аномалии", value=True, key="checkbox_anomali")
        
        with col2:
            st.session_state.data["year"] = st.session_state.data["timestamp"].dt.year
            min_year = int(st.session_state.data["year"].min())
            max_year = int(st.session_state.data["year"].max())
            year_range = st.slider("Диапазон лет", min_year, max_year, (min_year, max_year), key="year_slider_1")
        
        filtered = st.session_state.data[(st.session_state.data["city"].isin(selected_cities)) & (st.session_state.data["year"] >= year_range[0]) & (st.session_state.data["year"] <= year_range[1])]
        
        if st.session_state.results is not None:
            filtered_results = st.session_state.results[(st.session_state.results["city"].isin(selected_cities)) & (st.session_state.results["timestamp"].dt.year >= year_range[0]) & (st.session_state.results["timestamp"].dt.year <= year_range[1])]
        else:
            filtered_results = pd.DataFrame()
            show_anomalies = False

        if not filtered.empty and len(selected_cities) > 0:
            fig_lines = px.line(
                filtered,
                x="timestamp",
                y="temperature",
                color="city",
                title="Температуры по городам",
                labels={"temperature": "Температура", "timestamp": "Дата"})

            if show_anomalies and not filtered_results.empty and 'is_anomaly' in filtered_results.columns:
                anomalies = filtered_results[filtered_results["is_anomaly"]]
                if not anomalies.empty:
                    fig_lines.add_trace(go.Scatter(
                        x=anomalies["timestamp"],
                        y=anomalies["temperature"],
                        mode="markers",
                        name="Аномалии",
                        marker=dict(color="red", size=6, symbol="x"),
                        showlegend=True))
                    anomalies_count = filtered_results["is_anomaly"].sum()
                    st.write(f"Найдено аномалий: {anomalies_count}")
            st.plotly_chart(fig_lines, use_container_width=True)             

with tab_all3:
    st.subheader("Сезонные профили по городам")
    
    if st.session_state.data is not None and st.session_state.stat_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            cities_season = st.session_state.stat_df["city"].unique()
            selected_cities_season = st.multiselect("Выберите города для сравнения", options=cities_season, default=cities_season[:5])
        
        with col2:
            seasons = st.session_state.stat_df["season"].unique()
            selected_seasons = st.multiselect("Выберите сезоны", options=seasons,  default=seasons.tolist())
        
        filtered_1 = st.session_state.stat_df[(st.session_state.stat_df["city"].isin(selected_cities_season)) & (st.session_state.stat_df["season"].isin(selected_seasons))]
        
        if not filtered_1.empty:
            fig_grouped = px.bar(
                filtered_1,
                x="season",
                y="mean_temp",
                color="city",
                barmode="group",
                title="Сравнение сезонных средних по городам",
                labels={"mean_temp": "Средняя температура", "season": "Сезон"})
            st.plotly_chart(fig_grouped, use_container_width=True)
            

# визуализация для выбранного города
if st.session_state.city is not None and st.session_state.results is not None:
    st.header(f"Визуализация для города {st.session_state.city}")
    
    city_data = st.session_state.results[st.session_state.results["city"] == st.session_state.city]
    
    if not city_data.empty:
        tab_city1, tab_city2, tab_city3 = st.tabs(["Статистика города", "Временной ряд", "Сезонные профили"])
        
        with tab_city1:
            st.subheader(f"Статистика для {st.session_state.city}")
            
            desc_stats = city_data["temperature"].describe()
            st.write("Описательная статистика температуры:")
            st.write(desc_stats)
            
            fig_hist_city = px.histogram(
                city_data,
                x="temperature",
                title=f"Распределение температур в {st.session_state.city}",
                nbins=30)
            st.plotly_chart(fig_hist_city, use_container_width=True)
            
            fig_trend = px.line(
                city_data,
                x="timestamp",
                y="rolling_mean",
                title=f"Долгосрочный тренд температуры в {st.session_state.city}",
                labels={"rolling_mean": "Температура", "timestamp": "Дата"})
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab_city2:
            st.subheader(f"Временной ряд для {st.session_state.city}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                data_copy = st.session_state.data.copy()
                data_copy["year"] = data_copy["timestamp"].dt.year
                min_year = int(data_copy["year"].min())
                max_year = int(data_copy["year"].max())
                year_range = st.slider("Диапазон лет", min_year, max_year, (min_year, max_year), key="year_slider_2")
            
            with col2:
                show_anomalies = st.checkbox("Показать аномалии", value=True, key="checkbox_anomali_2")
                show_rolling = st.checkbox("Показать скользящее среднее", value=True, key="checkbox_rollong")
            
            filtered_city = city_data[(city_data["year"] >= year_range[0]) & (city_data["year"] <= year_range[1])]
            
            fig_city_lines = px.line(
                filtered_city,
                x="timestamp",
                y="temperature",
                title=f"Температура в {st.session_state.city}",
                labels={"temperature": "Температура", "timestamp": "Дата"})
            
            if show_rolling:
                fig_city_lines.add_trace(go.Scatter(
                    x=filtered_city["timestamp"],
                    y=filtered_city["rolling_mean"],
                    mode="lines",
                    name="Скользящее среднее (30 дней)",
                    line=dict(color="red", width=2)))
            
            if show_anomalies:
                anomalies_city = filtered_city[filtered_city["is_anomaly"]]
                if not anomalies_city.empty:
                    fig_city_lines.add_trace(go.Scatter(
                        x=anomalies_city["timestamp"],
                        y=anomalies_city["temperature"],
                        mode="markers",
                        name="Аномалии",
                        marker=dict(color="orange", size=8, symbol="x")))
            st.plotly_chart(fig_city_lines, use_container_width=True)
            

            if show_anomalies:
                anomalies_count = filtered_city["is_anomaly"].sum()
                st.write(f"Аномалий в выбранном диапазоне: {anomalies_count}")
        
        with tab_city3:
            st.subheader(f"Сезонные профили для {st.session_state.city}")
            if st.session_state.stat_df is not None:
                city_stat = st.session_state.stat_df[st.session_state.stat_df["city"] == st.session_state.city]
                
                table = city_stat[["season", "mean_temp", "std_temp"]].copy()
                table = table.round(2)
                table.columns = ["Сезон", "Средняя температура", "Стандартное отклонение"]
                table["Диапазон нормы"] = table.apply( lambda row: f"{row['Средняя температура'] - 2*row['Стандартное отклонение']:.1f} — {row['Средняя температура'] + 2*row['Стандартное отклонение']:.1f}", axis=1)
            
                st.dataframe(table)

                if not city_stat.empty:
                    fig_season_city = go.Figure()
                    
                    fig_season_city.add_trace(go.Bar(
                        x=city_stat["season"],
                        y=city_stat["mean_temp"],
                        name="Средняя температура",
                        error_y=dict(type="data", array=city_stat["std_temp"], visible=True)))
                    
                    fig_season_city.update_layout(
                        title=f"Сезонные профили в {st.session_state.city}",
                        xaxis_title="Сезон",
                        yaxis_title="Температура")
                    
                    st.plotly_chart(fig_season_city, use_container_width=True)
                    


# для синхронного запроса апи
def get_current_weather_sync(city_name, api_key):   
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key, "units": "metric"}
        
    response = requests.get(url, params=params, timeout=10)
        
    # ошибока
    if response.status_code == 401:
        return {"cod": 401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}
    
    if response.status_code == 200:
        data = response.json()
        return {"city": data["name"], "temperature": data["main"]["temp"], "cod": 200}
    else:
        return {"cod": response.status_code, "message": f"Ошибка {response.status_code}"}
        

# для асинхронного запроса
async def get_current_weather_async(city_name, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key, "units": "metric"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
            
        if response.status_code == 401:
            return {"cod": 401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}
            
        if response.status_code == 200:
            data = response.json()
            return {"city": data["name"], "temperature": data["main"]["temp"]}
        else:
            return {"error": f"Ошибка {response.status_code}"}
                

# для сравнения скорости запросов для одного города
def compare_sync_acync_methods(city_name, api_key):

    start = time.time()
    sync_res = get_current_weather_sync(city_name, api_key)
    sync_time = time.time() - start
    
    start = time.time()

    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {'q': city_name, 'appid': api_key, 'units': 'metric'}
        
    with httpx.Client() as client:
        response = client.get(url, params=params)
        
    async_time = time.time() - start
        
    if response.status_code == 200:
        data = response.json()
        async_res = {'temperature': data['main']['temp'], 'success': True}
    else:
        async_res = {"error": f"Ошибка {response.status_code}", "success": False}
            
    return {'sync': {'time': sync_time, 'result': sync_res}, 'async': {'time': async_time, 'result': async_res}}


# сравнение скорости асинхронногго и синхронного методов
st.sidebar.subheader("Сравнение синхронного и асинхронного методов запросов")

if st.sidebar.button("Сравнить скорость запросов"):
    
    if not st.session_state.city:
        st.sidebar.warning("Сначала выберите город")
    
    elif not st.session_state.api_key:
        st.sidebar.warning("Введите API-ключ для сравнения методов")
    
    else:
        with st.spinner("Идет сравнение..."):
            comp = compare_sync_acync_methods(st.session_state.city,  st.session_state.api_key)
            
            
            st.sidebar.write("Результаты сравнения:")
            st.sidebar.write(f"Синхронный: {comp['sync']['time']:.3f} сек")
            st.sidebar.write(f"Асинхронный: {comp['async']['time']:.3f} сек")
            
            if comp["async"]["time"] < comp["sync"]["time"]:
                diff = comp["sync"]["time"] - comp["async"]["time"]
                st.sidebar.write(f"Асинхронный метод быстрее на {diff:.3f} сек")
            else:
                diff = comp["async"]["time"] - comp["sync"]["time"]
                st.sidebar.write(f"Синхронный метод быстрее на {diff:.3f} сек")
            
            st.sidebar.write("Что лучше использовать?")
            st.sidebar.write("Для одного запроса разница минимальна. Асинхронность полезна при множестве запросов, но бесплатный API имеет лимит 1 запрос в секунду.")



# текущая температура и ее анализ
st.header("Анализ текущей температуры")

if not st.session_state.api_key:
    st.write("Введите API-ключ для получения текущей погоды")
    
elif not st.session_state.city:
    st.write("Выберите город в котором хотите узнать погоду")
    
else:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Получить и проанализировать текущую температуру"):
            with st.spinner("Получаем данные и анализируем их..."):
                result = get_current_weather_sync(st.session_state.city, st.session_state.api_key)
                
                if result.get("cod") == 200:
                    current_temp = result["temperature"]
                    st.session_state.current_temp = current_temp
                    
                    current_month = datetime.now().month
                    month_to_season = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn", 12: "winter"}
                    current_season = month_to_season[current_month]
                    
                    # исторические нормы для этого города и сезона
                    if st.session_state.stat_df is not None:
                        city_stat = st.session_state.stat_df[(st.session_state.stat_df["city"] == st.session_state.city) & (st.session_state.stat_df["season"] == current_season)]
    
                        if not city_stat.empty:
                            hist_mean = city_stat["mean_temp"].values[0]
                            hist_std = city_stat["std_temp"].values[0]
                            upper_bound = city_stat["upper_bound"].values[0]
                            lower_bound = city_stat["lower_bound"].values[0]
        
                            is_anomaly = (current_temp > upper_bound) or (current_temp < lower_bound)

                            # результаты
                            st.subheader("Результаты анализа: ")
                            st.write(f"Текущая температура в {st.session_state.city}: {current_temp:.1f} C")
                            st.write(f"Текущий сезон: {current_season}")
                            st.write(f"Историческая норма: {hist_mean:.1f} C (+-{2*hist_std:.1f} C)")
                            st.write(f"Диапазон нормы: {lower_bound:.1f} C - {upper_bound:.1f} C")

                            if not is_anomaly:
                                st.write(f"Температура в пределах нормы для сезона {current_season}")
                            else:
                                st.write(f"Температура аномальна для сезона {current_season}")
                                
                            fig_current = go.Figure()
                            
                            fig_current.add_trace(go.Scatter(
                                x=[current_season, current_season],
                                y=[lower_bound, upper_bound],
                                mode="lines",
                                line=dict(width=10, color="lightgreen"),
                                name="Норма"))

                            fig_current.add_trace(go.Scatter(
                                x=[current_season],
                                y=[hist_mean],
                                mode="markers",
                                marker=dict(size=15, color="green"),
                                name="Историческое среднее"))
                            
                            fig_current.add_trace(go.Scatter(
                                x=[current_season],
                                y=[current_temp],
                                mode="markers",
                                marker=dict(size=20, color="red" if is_anomaly else "blue", symbol="star"),
                                name="Текущая температура"))
                            
                            fig_current.update_layout(title=f"Сравнение текущей температуры с историческими данными",
                                xaxis_title="Сезон",
                                yaxis_title="Температура, C",
                                showlegend=True)
                            
                            st.plotly_chart(fig_current, use_container_width=True)
                            
                        else:
                            st.write(f"Нет исторических данных для {st.session_state.city} в сезон {current_season}")
                    else:
                        st.write("Сначала запустите анализ исторических данных")
                
                elif result.get("cod") == 401:
                    st.write(result.get("message"))
                else:
                    st.write(result.get("message"))
    # Справка
    with col2:
        st.subheader("Аномальность определяется следующим образом:")
        st.write("- Определяется текущий сезон")
        st.write("- Берутся исторические данные для выбранного города и сезона")
        st.write("- Вычисляется диапазон как среднее +- 2 * стандартное отклонение")
        st.write("- Если текущая температура вне диапазона, то она аномальна")
        










