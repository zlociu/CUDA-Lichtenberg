open System
open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.Hosting
open System.Runtime.InteropServices;
open Microsoft.AspNetCore.Http
open System.Text
open System.Drawing

type LichtenbergException(msg: string) = 
    inherit Exception(msg)

type LichtenbergQueryParameters(query: IQueryCollection) =
    let lightcolor = query["lightcolor"].ToString()
    let backcolor = query["backcolor"].ToString()
    let sizex = query["sizex"].ToString()
    let sizey = query["sizey"].ToString()
    let crateSize = query["cratesize"].ToString()
    let start = query["start"].ToString()
    let verticesCount = query["verticescount"].ToString()
    
    member val LightColor = Color.FromName lightcolor with get
    member val BackColor = Color.FromName backcolor with get
    member this.LightColorString = lightcolor
    member this.BackColorString = backcolor
    
    member val SizeX = Int32.Parse sizex with get
    member val SizeY = Int32.Parse sizey with get
    member val CrateSize = Int32.Parse crateSize with get
    member val Start = Int32.Parse start with get
    member val VerticesCount = Int32.Parse verticesCount with get

    static member GetParameterNames : string array =
        [| "lightcolor"; "backcolor"; "sizex"; "sizey"; "cratesize"; "start"; "verticescount" |]

    member this.CheckConstraints (vertexCnt: int) =
        this.SizeX < 64
        || this.SizeY < 64
        || this.CrateSize < 4
        || this.CrateSize > this.SizeX
        || this.CrateSize > this.SizeY
        || this.Start < 0
        || this.Start > this.SizeX * this.SizeY
        || this.VerticesCount < 1
        || this.VerticesCount > vertexCnt
    

module Lichtenberg = 
    [<DllImport("..\\LichtenbergGenerator\\x64\\Release\\LichtenbergGenerator.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)>]
    extern int createLichtenberg(int dimX, int dimY, int squareSize, int startPos, int verticesCount, string lightningColor, string backgroundColor, IntPtr xmlResult, int& lenght)


[<EntryPoint>]
let main args =
    let builder = WebApplication.CreateBuilder(args)
    let app = builder.Build()

    app.MapGet("/lichtenberg/{filename}",
        Func<HttpRequest, IResult>(
        fun (request) -> 
            try

                let filename = request.RouteValues["filename"].ToString()

                if String.IsNullOrEmpty filename then
                    raise (new LichtenbergException("Filename is missing"))

                for parameterName in LichtenbergQueryParameters.GetParameterNames do
                    if not (request.Query.ContainsKey(parameterName)) then
                        raise (new LichtenbergException($"{parameterName} is required"))

                let parameters = LichtenbergQueryParameters(request.Query)
                let vertexCnt = (parameters.SizeX * parameters.SizeY) / (parameters.CrateSize * parameters.CrateSize)

                if (parameters.CheckConstraints vertexCnt) then
                    raise (new LichtenbergException("Error: One or more query arguments are out of range"))
                            
                if parameters.LightColor.IsEmpty then
                    raise (new LichtenbergException("Error: lightcolor is not a valid color"))
                            
                if parameters.BackColor.IsEmpty then
                    raise (new LichtenbergException("Error: backcolor is not a valid color"))
                
                let pointer = Marshal.AllocHGlobal(350 + 100 * vertexCnt * sizeof<byte>)
                try
                    let mutable length = 0
                            
                    let status = Lichtenberg.createLichtenberg(
                        parameters.SizeX,
                        parameters.SizeY,
                        parameters.CrateSize,
                        parameters.Start,
                        parameters.VerticesCount,
                        parameters.LightColorString,
                        parameters.BackColorString,
                        pointer,
                        &length)

                    if status = 0 then
                        let xml = Marshal.PtrToStringAnsi(pointer)
                        Results.File(Encoding.UTF8.GetBytes(xml), "image/svg+xml", $"{filename}.svg", true)  
                    else
                        Results.BadRequest($"Error: generating Lichtenberg figure ended with status: {status}")
                finally
                    Marshal.FreeHGlobal(pointer)
            with
            | :? LichtenbergException as ex -> Results.BadRequest("Error: " + ex.Message)
            | _ -> Results.BadRequest("Request error")

            )) |> ignore

    app.Run()

    0 // Exit code