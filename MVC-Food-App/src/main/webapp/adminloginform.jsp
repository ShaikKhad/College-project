<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8" isELIgnored="false"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>

${msg}
<form action="validate" method="post">
Enter email : <input type="text" name="email">
Enter password : <input type="password" name="password">
<input type="submit">
</form>
</body>
</html>