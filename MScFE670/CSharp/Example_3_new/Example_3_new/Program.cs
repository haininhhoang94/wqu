using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Excel = Microsoft.Office.Interop.Excel;

namespace Example_3_new
{
    class Program
    {
        static void Main(string[] args)
        {
            Excel.Application app = new Excel.Application();
            app.Visible = true;
            app.Workbooks.Add();

            Excel._Worksheet currentSheet = app.ActiveSheet;
            currentSheet.Cells[1, "A"] = 2;
            var value = currentSheet.Cells[1, "A"].Value;
            Console.WriteLine(value);

            // prevent console from closing
            Console.ReadLine();
            /* app.Quit(); */
        }
    }
}
