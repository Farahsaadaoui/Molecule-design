<?php

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // retrieve the form data
    $username = $_POST['username'];
    $email = $_POST['email'];
    $password = $_POST['password'];

    // do something with the data
    // for example, store the data in a database

    // redirect the user to a success page
    header('Location: success.php');
    exit;
}

?>
