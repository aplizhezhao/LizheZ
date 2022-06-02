import Card from "./Card";
import ExpensesFilter from "./ExpensesFilter";
import ExpensesList from './ExpensesList'
import React, { useState } from "react";
import ExpensesChart from './ExpensesChart'
import "./Expenses.css";

function Expenses(props) {
  const [selectedYear, setSelectedYear] = useState("2022");

  const filterChangeHandler = (filterYear) => {
    setSelectedYear(filterYear); //Change selectedYear state to filterYear
  };

  const filteredExpenses = props.items.filter((expense) => {
    return expense.date.getFullYear().toString() === selectedYear;
  });

  return (
    <div>
      <Card className="expenses">
        <ExpensesFilter
          selected={selectedYear}
          onFilterChange={filterChangeHandler}
        />
        <ExpensesChart expenses={filteredExpenses}/>
        <ExpensesList filteredExpenses={filteredExpenses} onDelete={props.onDelete}/>
      </Card>
    </div>
  );
}

export default Expenses;
