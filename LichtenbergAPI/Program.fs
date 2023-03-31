open System
open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.Hosting
open System.Runtime.InteropServices;
open Microsoft.AspNetCore.Http
open System.Text

[<StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)>]
type InputArgs = 
    struct
        val startPos: int
        val VerticesCount: int
        val LightningColor: string
        val BackgroundColor: string

        new (startPos, verticesCount, lightningColor, backgroundColor) = 
            { 
                startPos = startPos
                VerticesCount = verticesCount
                LightningColor = lightningColor
                BackgroundColor = backgroundColor 
            }
    end

module Lichtenberg = 
    [<DllImport("..\\LichtenbergGenerator\\x64\\Release\\LichtenbergGenerator.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)>]
    extern int createLichtenberg(int dimX, int dimY, int crateSize, InputArgs arguments, IntPtr result, int& lenght)


[<EntryPoint>]
let main args =
    let builder = WebApplication.CreateBuilder(args)
    let app = builder.Build()

    app.MapGet("/lichtenberg/{filename}",
        Func<HttpRequest, IResult>(
        fun (request) -> 
            try
                let lightcolor = request.Query.["lightcolor"] |> string
                let backcolor = request.Query.["backcolor"] |> string
                let _sizex = request.Query.["sizex"] |> string
                let _sizey = request.Query.["sizey"] |> string
                let _cratesize = request.Query.["cratesize"] |> string 
                let _start = request.Query.["start"] |> string 
                let _verticescount = request.Query.["verticescount"] |> string           
                let filename = request.RouteValues.["filename"] |> string

                if filename = "" then
                    Results.BadRequest("Error: Filename is missing")
                elif lightcolor = "" || backcolor = "" || _sizex = "" || _sizey = "" || _cratesize = "" || _start = "" || _verticescount = "" then
                    Results.BadRequest("Error: One or more query arguments are missing")
                else
                    let sizex = Int32.Parse _sizex
                    let sizey = Int32.Parse _sizey
                    let cratesize = Int32.Parse _cratesize
                    let start = Int32.Parse _start
                    let verticescount = Int32.Parse _verticescount
                    let vertexCnt = (sizex * sizey) / (cratesize * cratesize)

                    if (sizex < 64 || sizey < 64 || 
                        cratesize < 4 || cratesize > sizex ||
                        cratesize > sizey || start < 0 || start > sizex * sizey ||
                        verticescount < 1 || verticescount > vertexCnt) then
                            Results.BadRequest("Error: One or more query arguments are out of range")
                    else
                        let inputArgs = InputArgs(start, verticescount, lightcolor, backcolor)
                        let pointer = Marshal.AllocHGlobal(350 + 100 * vertexCnt * sizeof<byte>)
                        let mutable length = 0
                        let status = Lichtenberg.createLichtenberg(sizex, sizey, cratesize, inputArgs, pointer, &length)

                        if status = 0 then
                            let xml = Marshal.PtrToStringAnsi(pointer)
                            Marshal.FreeHGlobal(pointer)
                            Results.File(Encoding.UTF8.GetBytes(xml), "text/xml", $"{filename}.xml", true)  
                        else
                            Marshal.FreeHGlobal(pointer)
                            Results.BadRequest($"Error: generating Lichtenberg figure ended with status: {status}")
            with
            | _ -> Results.BadRequest("Request error")

            )) |> ignore

    app.Run()

    0 // Exit code