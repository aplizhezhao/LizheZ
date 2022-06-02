import Expenses from "./components/Expenses";
import NewExpense from "./components/NewExpense";
import React, { useState } from "react";

const initialExpenses = [
  {
    id: "e1",
    title: "Car Insurance",
    amount: 297.46,
    date: new Date(2020, 1, 21),
  },
  {
    id: "e2",
    title: "Bookends",
    amount: 141.36,
    date: new Date(2022, 1, 24),
  },
  { id: "e3", title: "Grocery", amount: 23.5, date: new Date(2022, 1, 23) },
];

function App() {
  const [expenses, setExpenses] = useState(initialExpenses)

  const addExpenseHandler = (expense) => {
    setExpenses((prevExpenses) => {return [expense, ...prevExpenses]})
  };

  const deleteHandler = (selectedId) => {
    setExpenses((prevExpenses) => {return prevExpenses.filter((expense) => expense.id !== selectedId)})
  }

  return (
    <div>
      <Expenses items={expenses} onDelete={deleteHandler}/>
      <NewExpense onNewExpense={addExpenseHandler} />
    </div>
  );
}

export default App;
