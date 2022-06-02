import "./NewExpense.css";
import ExpenseForm from "./ExpenseForm";
import React, { useState } from "react";

const NewExpense = (props) => {
  const [isEditing, setIsEditing] = useState(false);

  const saveExpenseHandler = (enteredExpense) => {
    const expenseData = { ...enteredExpense, id: Math.random().toString() };
    props.onNewExpense(expenseData);
    setIsEditing(!isEditing)
  };

  const editingHandler = () => {
    setIsEditing(!isEditing);
  };

  return (
    <div className="new-expense">
      {!isEditing && <button onClick={editingHandler}>Add New Expense</button>}
      {isEditing && (
        <ExpenseForm
          onSaveExpense={saveExpenseHandler}
          onClick={editingHandler}
        />
      )}
    </div>
  );
};

export default NewExpense;
