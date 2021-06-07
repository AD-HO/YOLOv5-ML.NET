using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OnnxYoloV5
{
    class Program
        {

        const string modelPath = @"Assets\Models\yolov5s.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";
        static readonly string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        static void Main(string[] args)
        {
            


            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: 640, imageHeight: 640, resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "images", new[] { 1, 3, 640, 640 } },
                        { "561", new[] { 1, 25200, 85 } },
                    },
                    inputColumnNames: new[]
                    {
                        "images"
                    },
                    outputColumnNames: new[]
                    {
                        "561"
                    },
                    modelFile: modelPath));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV5BitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV5BitmapData, YoloV5Prediction>(model);

            // save model
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));

            foreach (string imageName in Directory.GetFiles(imageFolder))
            {
                using (var bitmap = new Bitmap(imageName))
                {
                    // predict
                    var predict = predictionEngine.Predict(new YoloV5BitmapData() { Image = bitmap });
                    var results = predict.GetResults(classesNames, 0.3f, 0.7f);

                    using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var res in results)
                        {
                            // draw predictions
                            var x1 = res.BBox[0];
                            var y1 = res.BBox[1];
                            var x2 = res.BBox[2];
                            var y2 = res.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
                                         new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }
                        var ss =Path.Combine(imageOutputFolder, Path.GetFileNameWithoutExtension(imageName)+"_Processed"+Path.GetExtension(imageName));
                        bitmap.Save(ss);
                        Console.WriteLine(Path.GetFileNameWithoutExtension(imageName) +" Processed ");
                    }
                }
            }
        }
    }
}
