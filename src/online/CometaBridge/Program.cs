using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Waveplus.DaqSys;
using Waveplus.DaqSysInterface;
using WaveplusLab.Shared.Definitions;


namespace CometaBridge
{
    internal class Program
    {
        /* 
        //Funciones aux
        //Notifica cambio de estado
        private static void StateChanged(object sender, DeviceStateChangedEventArgs e);
        //Notifica y recoge los datos de los sensores
        private static void DataAvailable(object sender, DataAvailableEventArgs e);
        //Imprime el estado de batería de un sensor
        private static void PrintSensorBattery(DataAvailableEventArgs e, int sensorNumber); 
        private static int DecodeBatteryPercentage(short sensorState);
        //Imprime las muestras de un sensor
        private static void PrintSensorSamples(DataAvailableEventArgs e, int sensorNumber, int maxSamplesToPrint); 
        //Iniciar el socket
        private static void StartTcpServer()
        //Envía un bloque de datos a Python
        private static void SendBlockToPython(DataAvailableEventArgs e);
        private static int GetSampleMatrixIndex(int sensorNumber, int positionInSelectedSensors, int sdkSensorCount);
        //Cerrar conexion
        private static void CloseTcp();
        */

        // Objeto principal control de la SDK 
        private static DaqSystem daqSystem;
        private static int receivedBlocks = 0;

        //TCP server
        private const int port = 5000;
        private static TcpListener tcpServer;
        private static TcpClient pythonClient;
        private static NetworkStream pythonStream;
        private static readonly object tcpLock = new object();

        private static readonly int[] selectedSensors = {2, 10, 12, 13, 15, 16};      


         private static void Main(string[] args)
        {
            Console.WriteLine("=== CometaBridge - SDK ===");

            try
            {
                // Crear objeto principal
                daqSystem = new DaqSystem();

                Console.WriteLine("DaqSystem creado correctamente.");
                Console.WriteLine("Estado inicial: " + daqSystem.State);
                Console.WriteLine();

                // Eventos
                daqSystem.StateChanged += StateChanged;
                daqSystem.DataAvailable += DataAvailable;

                Console.WriteLine("Esperando a que el dispositivo esté listo...");
                Console.WriteLine("Estado actual: " + daqSystem.State);
                Console.WriteLine();

                //Mostrar info de sensores
                Console.WriteLine("Sensores instalados: " + daqSystem.InstalledSensors);
                Console.WriteLine();

                if (daqSystem.InstalledSensors <= 0)
                {
                    Console.WriteLine("No se han detectado sensores instalados.");
                    Console.WriteLine("Pulsa una tecla para salir.");
                    Console.ReadKey();
                    return;
                }

                //Config sensores
                daqSystem.DisableSensor(0);

                Console.WriteLine("Configurando sensores como EMG...");
                SensorConfiguration sensorConfiguration = new SensorConfiguration();
                sensorConfiguration.SensorType = SensorType.EMG_SENSOR;
                foreach (int sensor in selectedSensors){
                    daqSystem.ConfigureSensor(sensorConfiguration, sensor);
                }    

                Console.WriteLine("Habilitando sensores");
                foreach (int sensor in selectedSensors){
                    daqSystem.EnableSensor(sensor);
                }

                //Config captura
                Console.WriteLine("Configurando captura a 2000 Hz");
                CaptureConfiguration captureConfiguration = new CaptureConfiguration();
                captureConfiguration.SamplingRate = SamplingRate.Hz_2000;
                captureConfiguration.ExternalTriggerEnabled = false;

                daqSystem.ConfigureCapture(captureConfiguration);

                Console.WriteLine("Configuración aplicada.");
                Console.WriteLine("Estado antes de capturar: " + daqSystem.State);
                Console.WriteLine();

                // Iniciar servidor TCP para comunicación con Python
                Console.WriteLine("Preparando conexión TCP con Python");
                StartTcpServer();

                //Iniciar captura
                Console.WriteLine("Iniciando captura, Pulsa ENTER para detener.");
                daqSystem.StartCapturing(DataAvailableEventPeriod.ms_25);
                Console.ReadLine();

                //Parar captura
                daqSystem.StopCapturing();

                CloseTcp();

                Console.WriteLine("Captura detenida.");
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine("ERROR:");
                Console.WriteLine(ex.GetType().FullName);
                Console.WriteLine(ex.Message);

                if (ex.InnerException != null)
                {
                    Console.WriteLine("Inner exception:");
                    Console.WriteLine(ex.InnerException.GetType().FullName);
                    Console.WriteLine(ex.InnerException.Message);
                }
            }

            Console.WriteLine();
            Console.WriteLine("Pulsa una tecla para cerrar.");
            Console.ReadKey();
        }

        ///////////////////////////////////FUNCIONES////////////////////////////////////////////////

        ///////////////////////////////////SDK///////////////////////////////////////////////////
        private static void StateChanged(object sender, DeviceStateChangedEventArgs e)
        {
            Console.WriteLine("[STATE CHANGED] Nuevo estado: " + daqSystem.State);
        }

        private static void DataAvailable(object sender, DataAvailableEventArgs e)
        {
            receivedBlocks++;

            if (e.Samples == null)
            {
                Console.WriteLine("e.Samples es null.");
                return;
            }

            SendBlockToPython(e);
        }

        /////////////////////////////////////////TCP/////////////////////////////////////////////////////////
        private static void StartTcpServer()
        {
            tcpServer = new TcpListener(IPAddress.Loopback, port);
            tcpServer.Start();

            Console.WriteLine("Esperando conexión de Python en 127.0.0.1:" + port);
            Console.WriteLine("Ejecuta ahora el receptor Python.");

            pythonClient = tcpServer.AcceptTcpClient();
            pythonStream = pythonClient.GetStream();

            Console.WriteLine("Python conectado correctamente.");
        }

        private static void SendBlockToPython(DataAvailableEventArgs e)
        {
            if (pythonStream == null || e.Samples == null)
            {
                return;
            }
            
            //Obtener dimensiones bloque
            int sdkSensorCount = e.Samples.GetLength(0);
            int nSamples = e.Samples.GetLength(1);
            int nChannelsToSend = selectedSensors.Length;

            StringBuilder sb = new StringBuilder();

            // Cabecera:
            // BLOCK idBloque numMuestras numCanales
            sb.Append("BLOCK ");
            sb.Append(receivedBlocks);
            sb.Append(" ");
            sb.Append(nSamples);
            sb.Append(" ");
            sb.Append(nChannelsToSend);
            sb.AppendLine();

            for (int sample = 0; sample < nSamples; sample++)//Bucle por muestras
            {
                for (int channel = 0; channel < nChannelsToSend; channel++)//Bucle por canales
                {
                    int sensorNumber = selectedSensors[channel];

                    int sampleMatrixIndex =
                        GetSampleMatrixIndex(sensorNumber, channel, sdkSensorCount);

                    if (sampleMatrixIndex < 0)
                    {
                        continue;
                    }

                    if (channel > 0)
                    {
                        sb.Append(",");
                    }

                    float value = e.Samples[sampleMatrixIndex, sample];

                    sb.Append(value.ToString("R", CultureInfo.InvariantCulture));
                }

                sb.AppendLine();
            }

            sb.AppendLine("END");

            // Convertir a bytes y enviar por TCP
            byte[] bytes = Encoding.UTF8.GetBytes(sb.ToString());

            lock (tcpLock)
            {
                try
                {
                    pythonStream.Write(bytes, 0, bytes.Length);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error enviando bloque a Python: " + ex.Message);
                }
            }
        }

        private static int GetSampleMatrixIndex(
            int sensorNumber,
            int positionInSelectedSensors,
            int sdkSensorCount
        )
        {
            int physicalIndex = sensorNumber - 1;

            if (physicalIndex >= 0 && physicalIndex < sdkSensorCount)
            {
                return physicalIndex;
            }

            if (positionInSelectedSensors >= 0 &&
                positionInSelectedSensors < sdkSensorCount)
            {
                return positionInSelectedSensors;
            }

            return -1;
        }

        private static void CloseTcp()
        {
            if (pythonStream != null)
            {
                pythonStream.Close();
            }

            if (pythonClient != null)
            {
                pythonClient.Close();
            }

            if (tcpServer != null)
            {
                tcpServer.Stop();
            }

            Console.WriteLine("Conexión TCP cerrada.");
        }
        
        //////////////////////////////////////AUX///////////////////////////////////////////////////////////
        private static void PrintSensorBattery(DataAvailableEventArgs e, int sensorNumber)
        {
            int sensorIndex = sensorNumber - 1;

            if (e.SensorStates == null)
            {
                Console.WriteLine("No hay información de estado/batería de sensores.");
                return;
            }

            int nSensors = e.SensorStates.GetLength(0);
            int nSamples = e.SensorStates.GetLength(1);

            if (sensorIndex < 0 || sensorIndex >= nSensors)
            {
                Console.WriteLine("Sensor " + sensorNumber + " fuera de rango.");
                return;
            }

            if (nSamples <= 0)
            {
                Console.WriteLine("No hay muestras de estado para el sensor " + sensorNumber + ".");
                return;
            }

            // Cogemos el estado asociado a la última muestra recibida de ese sensor.
            short sensorState = e.SensorStates[sensorIndex, nSamples - 1];

            int batteryPercentage = DecodeBatteryPercentage(sensorState);

            Console.WriteLine(
                "Batería sensor " + sensorNumber + ": " + batteryPercentage + "%"
            );
        }

        private static int DecodeBatteryPercentage(short sensorState)
        {
            // Nos quedamos solo con los dos bits menos significativos: BL1 BL0
            int batteryBits = sensorState & 0x0003;

            switch (batteryBits)
            {
                case 0:
                    return 0;

                case 1:
                    return 33;

                case 2:
                    return 66;

                case 3:
                    return 100;

                default:
                    return -1;
            }
        }

        private static void PrintSensorSamples(
            DataAvailableEventArgs e,
            int sensorNumber,
            int maxSamplesToPrint
        )
        {
            int nSensors = e.Samples.GetLength(0);
            int nSamples = e.Samples.GetLength(1);

            int sensorIndex = sensorNumber - 1;

            if (sensorIndex < 0 || sensorIndex >= nSensors)
            {
                Console.WriteLine(
                    "No puedo imprimir el sensor " + sensorNumber +
                    " porque e.Samples solo tiene " + nSensors + " canales."
                );

                return;
            }

            int samplesToPrint = Math.Min(nSamples, maxSamplesToPrint);

            Console.Write("Sensor SDK " + sensorNumber + ": ");

            for (int sample = 0; sample < samplesToPrint; sample++)
            {
                Console.Write(e.Samples[sensorIndex, sample].ToString("F2") + " ");
            }

            Console.WriteLine("uV");
        }
        

    }
}