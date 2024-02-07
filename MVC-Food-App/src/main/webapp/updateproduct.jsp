<%@page import="com.jsp.foodapp.dto.Product"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8" isELIgnored="false"%>

<%@ taglib prefix="form" uri="http://www.springframework.org/tags/form"%>

<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>

	<%
	Product product = (Product) request.getAttribute("com");
	%>

	<form:form action="updateproduct" modelAttribute="productinfo">

enter new product id   :        <form:input path="id" value="<%= product.getId() %>" readonly="true"/>	
	
enter new product name : 		<form:input path="name"  value="<%= product.getName() %>"/>

enter new product type : 		<form:input path="type" value="<%= product.getType() %>"/>

enter new product cost :		<form:input path="cost"  value="<%= product.getCost() %>"/>

<input type="submit">
	</form:form>
</body>
</html>