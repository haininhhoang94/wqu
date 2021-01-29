using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



namespace Module2_PRA
{
    /* Method Employee: */
    class Employee
    {
        // Constructor
        /* * 1. Constructor: */
        /* * Employee (int employeeId, String fullName, float salary, bool taxDeducted) */
        // protected for data hiding
        protected int employeeId;
        protected string fullName;
        protected float salary;
        protected bool taxDeducted;

        public Employee(int employeeId, string fullName, float salary, bool taxDeducted)
        {
            this.employeeId = employeeId;
            this.fullName = fullName;
            this.salary = salary;
            this.taxDeducted = taxDeducted;
        }

        /* * 2. Overloaded Constructor - If taxDeducted is not specified, assume it will */
        /* * be true with this overload constructor method: */
        /* * Employee(int employeeId, String fullName, float salary) */
        public Employee(int employeeId, string fullName, float salary)
        {
            this.employeeId = employeeId;
            this.fullName = fullName;
            this.salary = salary;
            this.taxDeducted = true;
        }

        /* * 3. getNetSalary - This method should return the Employee's salary minus 20% */
        /* * tax only if applicable */
        /* * float getNetSalary() */
        public float getNetSalary()
        {
            double result;
            if (this.taxDeducted == true)
            {
                result = salary * 0.8;
            }
            else
            {
                result = salary;
            }
            return Convert.ToSingle(result);
        }

        /* * 4. printInformation - This method should not return anything but rather */
        /* * print out the Employee's information in the format: */
        /* * "<employeeId>, <fullName> earns <net salary> per month" */
        /* * void printInformation() */
        public void printInformation()
        {
            Console.WriteLine(employeeId + ", " + fullName + " earns " + getNetSalary() + " per month");
        }
    }

    /* * Class WeeklyEmployee as subclass, inheriting from Employee */
    /* * 1. Constructor - Call a base class constructor */
    /* * WeeklyEmployee (int employeeId, String fullName, float salary, bool */
    /* * taxDeducted) */
    /* * 2. Overloaded Constructor - Call base */
    /* * WeeklyEmployee (int employeeId, String fullName, float salary) */
    class WeeklyEmployee : Employee
    {
        public WeeklyEmployee (int employeeId, string fullName, float salary, bool taxDeducted) 
            : base(employeeId, fullName, salary, taxDeducted){}
        public WeeklyEmployee (int employeeId, string fullName, float salary) 
            : base(employeeId, fullName, salary){}
        /* * 3. getNetSalary - Return the employee's weekly salary minus 20% tax. In the */
        /* * base class it is monthly, so this is different and should OVERRIDE the main */
        /* * method */
        public new float getNetSalary()
        {
            double weekly_salary;
            weekly_salary = salary / 4.5; // assume there is 4.5 weeks per month
            double result;
            if (this.taxDeducted == true)
            {
                result = weekly_salary * 0.8;
            }
            else
            {
                result = weekly_salary;
            }
            return Convert.ToSingle(result);
        }
        /* * 4. printInformation - print: */
        /* * "<employeeId>, <fullName> earns <netWeeklySalary> per week" */
        /* * void printInformation() */
        /* * */ 
        public new void printInformation()
        {
            Console.WriteLine(employeeId + ", " + fullName + " earns " + getNetSalary() + " per week");
        }
    }

}
