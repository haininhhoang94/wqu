using System;
using System.Collections.Generic;
using System.Text;

namespace Example_3
{
    class Customer
    {
       int customerId;
       string fullName;

       public Customer(int customerId, string fullName)
       {
           this.customerId = customerId;
           this.fullName = fullName;
       }
    }

    class Sale
    {
        int saleId;
        float saleAmount;
        int customerId;
        
        public Sale(int saleId, float saleAmount, int customerId)
        {
            this.saleId = saleId;
            this.saleAmount = saleAmount;
            this.customerId = customerId;
        }
    }

    static List<Customer> customers = new List<Customer>();
    static List<Sale> sales = new List<Sale>();

    staic
}
