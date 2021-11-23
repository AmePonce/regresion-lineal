
async function getData() {

  // Vamos a obtener los datos a traves del fetch
    const datosCasasR = await fetch('https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json');
    const datosCasas = await datosCasasR.json();

    //solo vamos a extraer el numero de cuarto y el precio de la casa
    const datosLimpios = datosCasas.map(casa => ({
      precio: casa.Precio,
      cuartos: casa.NumeroDeCuartosPromedio
    }))

    //Vamos a evitar que existan datos nulos 
    .filter(casa => (casa.precio != null && casa.cuartos != null));

    return datosLimpios;
  }
  // Vamos a ayudar a dezplegarlo en la pantalla y que se mapeé a "x" y "y"
  function visualizarDatos(data){
    const valores = data.map(d => ({
      x: d.cuartos,
      y: d.precio,
    }));

    tfvis.render.scatterplot(
      {name: 'Cuartos vs Precio'},
      {values: valores},
      {
        xLabel: 'Cuartos',
        yLabel: 'Precio',
        height: 300
      }
    );
  }

// Declararemos al secuencial que hace que los datos fluyan de un tensor al siguiente 
function crearModelo(){
  const modelo = tf.sequential();

  // Vamos a agregar la capa oculta que va a recibir 1er dato
  modelo.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Ahora agregaremos una capa de salida que va a tener 1 sola unidad
  modelo.add(tf.layers.dense({units: 1, useBias: true}));

  return modelo;
}

// Tendremos a adam, que es un optimizador predefinido
const optimizador = tf.train.adam()

// Vamos a predefinir la libreria de losses  (mide los  puntos de los modelos sin entrenar)
const funcion_perdida = tf.losses.meanSquaredError;

//mins squere error 
const metricas = ['mse'];

async function entrenarModelo(model, inputs, labels) {
  
  model.compile({
    optimizer: optimizador,
    loss: funcion_perdida,
    metrics: metricas,
  });

 // Ahora pondremos como se va a ir entrenando el modelo
  const surface = { name: 'show.history live', tab: 'Training' };

  // El numero de registrso que va incluyendo
  const tamanioBatch = 28;

  //Las vueltas totales que tiene
  const epochs = 50;
  // Mantendremos las metricas para poder graficarlas
  const history = [];

  return await model.fit(inputs, labels, {
    tamanioBatch,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}


// Ahora convertiremos los datos porque tensor flow los prefiere en 0 y 1
function convertirDatosATensores(data){
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const entradas = data.map(d => d.cuartos)
    const etiquetas = data.map(d => d.precio);

    //tensor tendra dos dimenciones
    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1]);
    const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1]);

    //Ocuparemos lo siguiente para desregularizar 
    const entradasMax = tensorEntradas.max();
    const entradasMin = tensorEntradas.min();
    const etiquetasMax = tensorEtiquetas.max();
    const etiquetasMin = tensorEtiquetas.min();

    // Tendremos nuestros (dato -min) / (max-min) entrada para normalizar
    const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin));
    const etiquetasNormalizadas = tensorEtiquetas.sub(etiquetasMin).div(etiquetasMax.sub(etiquetasMin));

      return {
        entradas: entradasNormalizadas,
        etiquetas: etiquetasNormalizadas,
        
        entradasMax,
        entradasMin,
        etiquetasMax,
        etiquetasMin,
      }
      //Nuestra funcion tidy tiene el objetivo que una vwz creada los tensores de desase de aquellas que no son utiles

  });
}

// Se iran probando las funciones 
async function run() {

    const data = await getData();

    visualizarDatos(data);

    modelo = crearModelo();

    const tensorData = convertirDatosATensores (data);
    const {entradas, etiquetas} = tensorData;
    await entrenarModelo(modelo, entradas, etiquetas)


}
run();
