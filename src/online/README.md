# CometaBridge

Programa puente en C# encargado de comunicarse con el sistema COMETA/Waveplus mediante la SDK oficial y enviar las muestras EMG al pipeline Python a través de TCP.

El puente se compila y ejecuta de forma independiente del entorno virtual.

## Requisitos

* Windows.
* Driver USB oficial de COMETA/Waveplus.
* .NET SDK para disponer del comando `dotnet`.
* .NET Framework 4.6.1 Developer Pack.
* SDK oficial `Waveplus.Daq SDK 5.0.7`.

El entorno de desarrollo no es obligatorio. Puede utilizarse Visual Studio Code, Visual Studio o únicamente PowerShell.

## Librerías de COMETA

Las librerías utilizadas se encuentran en la carpeta `NET.4.6.1` de la SDK oficial de COMETA:

```text
CyUSB.dll
Waveplus.DaqSys.dll
Waveplus.DaqSysInterface.dll
Waveplus.DaqSys.pdb
Waveplus.DaqSysInterface.pdb
```

Las DLL permiten acceder al dispositivo y a las funciones de adquisición. Los archivos `.pdb` contienen información de depuración y son opcionales.

## Estructura

```text
online/
└── CometaBridge/
    ├── CometaBridge.csproj
    ├── Program.cs
    └── lib/
        ├── CyUSB.dll
        ├── Waveplus.DaqSys.dll
        ├── Waveplus.DaqSys.pdb
        ├── Waveplus.DaqSysInterface.dll
        └── Waveplus.DaqSysInterface.pdb
```

## Configuración del proyecto

El proyecto está configurado para:

```text
.NET Framework 4.6.1
Arquitectura x86
```

El archivo `CometaBridge.csproj` contiene las referencias a las DLL mediante rutas relativas:

La arquitectura `x86` deberá confirmarse durante las pruebas con el dispositivo real.

## Compilación

Desde la carpeta del proyecto:

```powershell
cd online\CometaBridge
dotnet build
```

El ejecutable se genera en:

```text
bin\Debug\net461\CometaBridge.exe
```

## Ejecución

Puede ejecutarse mediante:

```powershell
dotnet run
```

o directamente:

```powershell
.\bin\Debug\net461\CometaBridge.exe
```

## Relación con Python

El puente C# y el pipeline Python son procesos independientes.

```text
COMETA → Bridge C# → TCP local → Pipeline Python
```
Los sockets TCP no requieren librerías externas:

```csharp
using System.Net;
using System.Net.Sockets;
```

```python
import socket
```

## Reproducción en otro ordenador

1. Instalar el driver oficial de COMETA.
2. Instalar .NET SDK.
3. Instalar .NET Framework 4.6.1 Developer Pack.
4. Copiar las DLL de la SDK en `online/CometaBridge/lib/`.
5. Ejecutar:

```powershell
dotnet build
```

6. Ejecutar el puente y el pipeline Python en terminales separadas.

