using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArrayExample
{
   class Program
   {
      static void Main(string[] args)
      {
         Console.WriteLine("Using an array:");
         int[] arr = new int[3];
         arr[0] = 1;
         arr[1] = 3;
         arr[2] = 5;
         Console.WriteLine("" + arr[0] + ", " + arr[1] + ", " + arr[2]);
         Console.WriteLine();
      }
   }
}
