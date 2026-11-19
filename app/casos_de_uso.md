# Casos de uso para la demo

## 1. Casos claros: los cuatro modelos coinciden

**Sports**
```
Lakers beat Celtics 112-104 in overtime as LeBron James scores 38 points in the season opener.
```
Sports en los cuatro (99 % / 90 % / 86 % / 54 %).

**Business**
```
Wall Street stocks tumble as oil prices surge and the Federal Reserve signals another interest rate hike.
```
Business en los cuatro (100 % / 85 % / 92 % / 99 %).

**Sci/Tech**
```
Researchers unveil a new AI chip that accelerates machine learning on smartphones while cutting power use.
```
Sci/Tech en los cuatro (82 % / 90 % / 90 % / 98 %).

**World**
```
UN envoy meets rebel leaders in Geneva to negotiate a ceasefire after weeks of fighting along the border.
```
World en los cuatro (98 % / 91 % / 94 % / 100 %).

## 2. Casos de frontera: la categoría es discutible, pero la respuesta es razonable

**Empresa tecnológica con noticia financiera**
```
Microsoft shares fall after the software giant reports weaker than expected quarterly cloud revenue.
```
TF-IDF responde Business (82 %); BiLSTM, NewsHybridNet y DistilBERT responden Sci/Tech (73 % / 86 % / 79 %).
Sirve para explicar la confusión Business / Sci/Tech, la más frecuente en el test.

**Salida a bolsa de una empresa tecnológica**
```
Google sets price range for its long-awaited IPO, raising up to $3 billion on Wall Street.
```
Business en los cuatro (94 % / 88 % / 86 % / 83 %). Ganan los términos financieros sobre el nombre de la empresa.

**Ciencia con impacto mundial**
```
Scientists warn that rising sea levels could displace millions of people in coastal Asia by 2050.
```
Sci/Tech en los cuatro (62 % / 51 % / 83 % / 76 %). La confianza baja en las líneas base, porque el tema también es World.

## 3. Casos fuera de distribución que se resuelven bien

**Política británica (estilo BBC)**
```
The Prime Minister has announced a new bill on immigration which will be debated in the House of Commons next week.
```
World en los cuatro (85 % / 88 % / 90 % / 100 %). Es el mapeo `politics → World` usado con BBC News.

**Tema reciente que no existía en AG News (2004)**
```
OpenAI launches a new chatbot model that writes code and passes bar exams, raising concerns among regulators.
```
Sci/Tech en los cuatro (81 % / 68 % / 82 % / 99 %). DistilBERT es el más seguro; las líneas base acertaron igualmente.

## Cómo usarlos

1. Empieza con los cuatro de la sección 1: muestran que todos los modelos funcionan.
2. Pasa a la sección 2 para enseñar que las categorías no siempre son nítidas y que las barras de probabilidad lo reflejan.
3. Cierra con la sección 3 para hablar de generalización a fuentes y épocas no vistas.
4. En NewsHybridNet, mira el mapa de atención y la compuerta de fusión bajo los resultados.
