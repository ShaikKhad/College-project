<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8" isELIgnored="false"%>
    <%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
<table align="center" cellPadding="20px" border="1">
		<th>id</th>
		<th>name</th>
		<th>type</th>
		<th>cost</th>
        <th>Add</th>

		<c:forEach var="product" items="${productlist}">
        <tr>
        <td> ${product.getId() } </td> 
        <td> ${product.getName() } </td>
        <td> ${product.getType() } </td>
        <td> ${product.getCost() } </td>
        <td> <a href="additem?id=${product.getId()}">add</a> </td>
        
        </tr>
		</c:forEach>





	</table>
	
	<a href="useroptions.jsp">Back</a>                                           <a href="savefoodorders">Submit</a>
 	
</body>
</html>