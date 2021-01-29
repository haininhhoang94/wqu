using System;
using Excel = Microsoft.Office.Interop.Excel;
using System.Collections.Generic;
using System.Linq;

namespace WorldQuant_Module3_CSA_SkeletonCode
{
    class Program
    {
        static Excel.Workbook workbook;
        static Excel.Application app;

        static void Main(string[] args)
        {
            app = new Excel.Application();
            app.Visible = true;
            try
            {
                workbook = app.Workbooks.Open("property_pricing.xlsx", ReadOnly: false);
            }
            catch
            {
                SetUp();
            }

            var input = "";
            while (input != "x")
            {
                PrintMenu();
                input = Console.ReadLine();
                try
                {
                    var option = int.Parse(input);
                    switch (option)
                    {
                        case 1:
                            try
                            {
                                Console.Write("Enter the size: ");
                                var size = float.Parse(Console.ReadLine());
                                Console.Write("Enter the suburb: ");
                                var suburb = Console.ReadLine();
                                Console.Write("Enter the city: ");
                                var city = Console.ReadLine();
                                Console.Write("Enter the market value: ");
                                var value = float.Parse(Console.ReadLine());

                                AddPropertyToWorksheet(size, suburb, city, value);
                            }
                            catch
                            {
                                Console.WriteLine("Error: couldn't parse input");
                            }
                            break;
                        case 2:
                            Console.WriteLine("Mean price: " + CalculateMean());
                            break;
                        case 3:
                            Console.WriteLine("Price variance: " + CalculateVariance());
                            break;
                        case 4:
                            Console.WriteLine("Minimum price: " + CalculateMinimum());
                            break;
                        case 5:
                            Console.WriteLine("Maximum price: " + CalculateMaximum());
                            break;
                        default:
                            break;
                    }
                } catch { }
            }

            // save before exiting
            workbook.Save();
            workbook.Close();
            app.Quit();
        }

        static void PrintMenu()
        {
            Console.WriteLine();
            Console.WriteLine("Select an option (1, 2, 3, 4, 5) " +
                              "or enter 'x' to quit...");
            Console.WriteLine("1: Add Property");
            Console.WriteLine("2: Calculate Mean");
            Console.WriteLine("3: Calculate Variance");
            Console.WriteLine("4: Calculate Minimum");
            Console.WriteLine("5: Calculate Maximum");
            Console.WriteLine();
        }

        static void SetUp()
        {
            // Setup the worksheet when the application is launched for
            // the first time (create a new workbook titled
            // "property_pricing.xlas" if the workbook is not exist yet)
            app.Workbooks.Add();
            workbook = app.ActiveWorkbook;
            
            // in the problem, there is no "template" to store the properties,
            // so it should create some form of template that store the new
            // properties in each row. This should only be doing when there is
            // no existing xlsx file exist
            Excel._Worksheet cw = app.ActiveSheet;
            cw.Cells[1, "A"] = "Size";
            cw.Cells[1, "B"] = "Surburb";
            cw.Cells[1, "C"] = "City";
            cw.Cells[1, "D"] = "Market value";
            workbook.SaveAs("property_pricing.xlsx");
            Console.WriteLine("Initializing complete!");
        }

        static void AddPropertyToWorksheet(float size, string suburb, string city, float value)
        {
            // Implement this method
            /* AddPropertyToWorksheet(size, suburb, city, value); */
            // In this method, we need to read the current data available first
            // before insert our new data
            
            // Read current data
            Excel._Worksheet cw = app.ActiveSheet;
            int i = 2; // from first cell
            while (true)
            {
                if (cw.Cells[i, "A"].value == null)
                {
                    // set the value and volla
                    cw.Cells[i, "A"] = size;
                    cw.Cells[i, "B"] = suburb;
                    cw.Cells[i, "C"] = city;
                    cw.Cells[i, "D"] = value;
                    break;
                }
                else
                {
                    i++;
                }
            }
        }

        static float CalculateMean()
        {
            // In order to calculate mean, we need to read the market
            // values of the properties first
            List<double> mv = new List<double>();
            mv = market_values();
            // Calculate Average by LINQ
            return Convert.ToSingle(mv.Average());
        }

        static float CalculateVariance()
        {
            // In order to calculate mean, we need to read the market
            // values of the properties first
            List<double> mv = new List<double>();
            mv = market_values();

            // Initalize result
            float result;

            // Calculate sample variance
            int n = mv.Count();
            float M2 = 0;
            foreach (float x in mv)
            {
                float delta = x - CalculateMean();
                M2 += delta * delta;
            }
            result = M2 / (n - 1);

            return result;
        }

        static float CalculateMinimum()
        {
            // In order to calculate mean, we need to read the market
            // values of the properties first
            List<double> mv = new List<double>();
            mv = market_values();
            return Convert.ToSingle(mv.Min());
        }

        static float CalculateMaximum()
        {
            // In order to calculate mean, we need to read the market
            // values of the properties first
            List<double> mv = new List<double>();
            mv = market_values();
            return Convert.ToSingle(mv.Max());
        }

        static List<double> market_values()
        {
            Excel._Worksheet cw = app.ActiveSheet;

            // New list for result
            List <double> result = new List<double>();

            // Read current data
            int i = 2; // from first cell
            while (true)
            {
                if (cw.Cells[i, "A"].value == null)
                {
                    break;
                }
                else
                {
                    // Add the value and continue the loop
                    result.Add(cw.Cells[i, "D"].value);
                    i++;
                }
            }
            return result;
        }
    }
}
